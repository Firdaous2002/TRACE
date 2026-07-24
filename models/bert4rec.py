from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import attention


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        max_len: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.mask_token = num_items + 1

        self.item_emb = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.emb_norm = nn.LayerNorm(hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, num_items + 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight.data[1:])
        nn.init.xavier_uniform_(self.pos_emb.weight.data)

    def _make_attention_mask(
        self,
        input_seq: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pad_mask is None:
            pad_mask = (input_seq == 0)

        B = pad_mask.size(0)
        attn_mask = torch.zeros((B, 1, 1, pad_mask.size(1)), device=input_seq.device)
        attn_mask = attn_mask.masked_fill(pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        # Unmask all-padding query rows so softmax never sees an all-(-inf) row (NaN-safety).
        row_all_pad = pad_mask.all(dim=-1).view(B, 1, 1, 1)
        attn_mask = attn_mask.masked_fill(row_all_pad, 0.0)
        return attn_mask

    def _prepare_prediction_input(self, input_seq: torch.Tensor) -> torch.Tensor:
        has_mask = (input_seq == self.mask_token).any(dim=1)
        if has_mask.all():
            return input_seq

        prepared = input_seq.clone()
        rows = ~has_mask
        prepared[rows, :-1] = input_seq[rows, 1:]
        prepared[rows, -1] = self.mask_token
        return prepared

    def _mask_indices(self, input_seq: torch.Tensor) -> torch.Tensor:
        mask_positions = (input_seq == self.mask_token)
        has_mask = mask_positions.any(dim=1)
        if (~has_mask).any():
            raise ValueError('BERT4Rec.predict expected at least one [MASK] token per sequence')
        L = input_seq.size(1)
        mask_indices = L - 1 - mask_positions.flip(1).to(torch.int64).argmax(dim=1)
        return mask_indices

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        _, L = input_seq.shape
        device = input_seq.device

        seq_emb = self.item_emb(input_seq)
        positions = torch.arange(L, device=device).unsqueeze(0)
        seq_emb = seq_emb + self.pos_emb(positions)
        seq_emb = self.emb_norm(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)

        pad_mask = (input_seq == 0)
        seq_emb = seq_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        attn_mask = self._make_attention_mask(input_seq, pad_mask)

        for layer in self.layers:
            seq_emb = layer(seq_emb, attn_mask, pad_mask)

        seq_emb = self.final_norm(seq_emb)
        seq_emb = seq_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return seq_emb

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        masked_seq = self._prepare_prediction_input(input_seq)
        hidden = self.forward(masked_seq)
        mask_indices = self._mask_indices(masked_seq)
        batch_idx = torch.arange(masked_seq.size(0), device=masked_seq.device)
        mask_hidden = hidden[batch_idx, mask_indices, :]
        return self.output_proj(mask_hidden)

    def forward_from_emb(self, seq_emb: torch.Tensor, input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_emb = self.emb_norm(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)

        if input_seq is not None:
            pad_mask = (input_seq == 0)
            attn_mask = self._make_attention_mask(input_seq, pad_mask)
        else:
            pad_mask = (seq_emb.abs().sum(dim=-1) == 0)
            attn_mask = None

        for layer in self.layers:
            seq_emb = layer(seq_emb, attn_mask, pad_mask)

        seq_emb = self.final_norm(seq_emb)
        seq_emb = seq_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        hidden = seq_emb[:, -1, :]
        if input_seq is not None:
            has_mask = (input_seq == self.mask_token).any(dim=1)
            if has_mask.any():
                mask_indices = self._mask_indices(input_seq[has_mask])
                batch_idx = torch.arange(input_seq.size(0), device=input_seq.device)[has_mask]
                hidden = hidden.clone()
                hidden[has_mask] = seq_emb[batch_idx, mask_indices, :]

        return self.output_proj(hidden)

    def compute_loss(
        self,
        masked_input: torch.Tensor,
        labels: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.forward(masked_input)
        loss_mask = mask_positions.bool() if mask_positions is not None else (labels > 0)
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=masked_input.device, requires_grad=True)

        logits = self.output_proj(hidden[loss_mask])
        targets = labels[loss_mask]
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def top_k(self, input_seq: torch.Tensor, k: int = 10) -> torch.Tensor:
        scores = self.predict(input_seq).float()
        scores[:, 0] = -float('inf')
        seen_items = input_seq.masked_fill(input_seq > self.num_items, 0)
        scores.scatter_(1, seen_items, -float('inf'))
        _, topk_ids = scores.topk(k, dim=-1)
        return topk_ids


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(hidden_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, x_norm, x_norm, mask=attn_mask)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        x = self.dropout(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)
        query, key, value = [
            proj(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
            for proj, x in zip(self.linear_layers, (query, key, value))
        ]

        dropout_p = self.dropout.p if self.training else 0.0
        attn_mask = mask if mask is not None else None
        # BERT4Rec uses an additive (-inf) attention mask. On our CUDA setup,
        # the fused SDPA kernels can segfault under AMP for explain-time batched
        # inference, while the math kernel is stable.
        use_safe_math_sdpa = (
            query.is_cuda
            and attn_mask is not None
            and query.dtype in {torch.float16, torch.bfloat16}
        )
        if use_safe_math_sdpa:
            with attention.sdpa_kernel(attention.SDPBackend.MATH):
                x = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                )
        else:
            x = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
            )
        x = x.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
