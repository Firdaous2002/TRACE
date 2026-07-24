"""
SASRec: Self-Attentive Sequential Recommendation.

Unidirectional (causal) Transformer for sequential recommendation.
Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    """
    SASRec model with causal self-attention.

    Args:
        num_items: total number of items (item IDs: 1..num_items, 0=padding)
        hidden_dim: embedding dimension
        max_len: maximum sequence length
        num_heads: number of attention heads
        num_layers: number of transformer blocks
        dropout: dropout rate
    """

    _causal_mask_cache: torch.Tensor

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

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # persistent=False keeps this out of state_dict (checkpoint-compatible).
        self.register_buffer(
            "_causal_mask_cache",
            torch.tril(torch.ones((max_len, max_len), dtype=torch.bool)),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight.data[1:])
        nn.init.xavier_uniform_(self.pos_emb.weight.data)

    def _make_attention_mask(
        self,
        input_seq: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, L = input_seq.shape

        if pad_mask is None:
            pad_mask = (input_seq == 0)

        causal = self._causal_mask_cache[:L, :L]
        key_is_valid = (~pad_mask).unsqueeze(1)
        mask = causal.unsqueeze(0) & key_is_valid

        # Unmask all-padding query rows so softmax never sees an all-False row (NaN-safety).
        row_is_pad = pad_mask.unsqueeze(-1)
        mask = mask | row_is_pad

        return mask

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        B, L = input_seq.shape
        device = input_seq.device

        # Embedding with sqrt(d) scaling (following Kang's original)
        seq_emb = self.item_emb(input_seq)
        seq_emb = seq_emb * (self.hidden_dim ** 0.5)
        positions = torch.arange(L, device=device).unsqueeze(0)
        seq_emb = seq_emb + self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        pad_mask = (input_seq == 0)
        seq_emb = seq_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        attn_mask = self._make_attention_mask(input_seq, pad_mask)

        for layer in self.layers:
            seq_emb = layer(seq_emb, attn_mask, pad_mask)

        seq_emb = self.final_norm(seq_emb)
        return seq_emb

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for all items based on the last position.

        Args:
            input_seq: (B, L) padded item ID sequence
        Returns:
            scores: (B, num_items+1) raw scores for each item
        """
        hidden = self.forward(input_seq)  # (B, L, D)
        last_hidden = hidden[:, -1, :]  # (B, D)
        scores = last_hidden @ self.item_emb.weight.T  # (B, num_items+1)
        return scores

    def forward_from_emb(self, seq_emb: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward from pre-computed embeddings (matches predict exactly).
        Args:
            seq_emb: (B, L, D) item embeddings (with sqrt(d) scaling + pos_emb)
            input_seq: (B, L) item IDs (for proper attention mask)
        Returns:
            scores: (B, num_items+1) raw scores for each item
        """
        pad_mask = (input_seq == 0)
        attn_mask = self._make_attention_mask(input_seq, pad_mask)
        for layer in self.layers:
            seq_emb = layer(seq_emb, attn_mask, pad_mask)
        seq_emb = self.final_norm(seq_emb)
        last_hidden = seq_emb[:, -1, :]
        scores = last_hidden @ self.item_emb.weight.T
        return scores

    def compute_loss(
        self,
        input_seq: torch.Tensor,
        target_pos: torch.Tensor,
        target_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        CE (full softmax) loss for sequence prediction.

        At each position t, compute cross-entropy over all items
        with the actual next item as the target.

        Args:
            input_seq: (B, L) input sequence
            target_pos: (B, L) positive targets (next items)
            target_neg: ignored (kept for interface compatibility)
        Returns:
            loss: scalar CE loss
        """
        hidden = self.forward(input_seq)  # (B, L, D)

        # Full softmax over all items at every position
        logits = hidden @ self.item_emb.weight.T  # (B, L, num_items+1)

        # Only compute loss at non-padding target positions
        mask = (target_pos != 0)
        logits_flat = logits[mask]      # (N, num_items+1)
        target_flat = target_pos[mask]  # (N,)

        loss = F.cross_entropy(logits_flat, target_flat)
        return loss

    @torch.no_grad()
    def top_k(self, input_seq: torch.Tensor, k: int = 10) -> torch.Tensor:
        scores = self.predict(input_seq).float()
        scores[:, 0] = -float('inf')
        scores.scatter_(1, input_seq, -float('inf'))
        _, topk_ids = scores.topk(k, dim=-1)
        return topk_ids


class TransformerBlock(nn.Module):
    """Transformer block with custom multi-head attention (no NaN gradients)."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        # Pre-norm: LayerNorm BEFORE attention (matching Kang's original)
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, L, H, hd).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, L, H, hd).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, L, H, hd).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask.unsqueeze(1),
            dropout_p=dropout_p,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)
        x = x + self.dropout(attn_out)

        # Pre-norm: LayerNorm BEFORE FFN
        x = x + self.ffn(self.norm2(x))
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x
