"""
GRU4Rec: Session-based Recommendations with Recurrent Neural Networks.

GRU-based sequential recommendation model.
Reference: Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks", ICLR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GRU4Rec(nn.Module):
    """
    GRU4Rec model.

    Architecture:
    - Item embedding
    - Single-layer GRU
    - Dot product with item embeddings for scores

    Args:
        num_items: total number of items (1..num_items, 0=padding)
        emb_dim: item embedding dimension
        hidden_dim: GRU hidden dimension (typically 2x emb_dim)
        max_len: maximum sequence length
        dropout: dropout rate
        num_gru_layers: number of GRU layers
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 128,
        max_len: int = 50,
        dropout: float = 0.3,
        num_gru_layers: int = 1,
        emb_dim: int = 64,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.num_gru_layers = num_gru_layers

        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim, emb_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight.data[1:])
        nn.init.xavier_uniform_(self.dense.weight)

    def init_hidden(self, batch_size: int, device: str | torch.device) -> torch.Tensor:
        return torch.zeros(self.num_gru_layers, batch_size, self.hidden_dim, device=device)

    def _project_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.output_dropout(hidden_state)
        emb_out = self.dense(hidden_state)
        return emb_out @ self.item_emb.weight.T

    def encode_step(self, input_items: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        if input_items.dim() == 1:
            input_items = input_items.unsqueeze(1)
        seq_emb = self.item_emb(input_items)
        seq_emb = self.emb_dropout(seq_emb)
        _, hidden_state = self.gru(seq_emb, hidden)
        return hidden_state[-1], hidden_state

    def score_items(self, hidden_state: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        hidden_state = self.output_dropout(hidden_state)
        emb_out = self.dense(hidden_state)
        item_emb = self.item_emb(item_ids)
        if item_ids.dim() == 1:
            return (emb_out * item_emb).sum(dim=-1)
        return (emb_out.unsqueeze(1) * item_emb).sum(dim=-1)

    def step(self, input_items: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        step_hidden, hidden = self.encode_step(input_items, hidden)
        logits = self._project_hidden(step_hidden)
        return logits, hidden

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns GRU hidden states at all positions.

        Args:
            input_seq: (B, L) padded item ID sequence (left-padded with 0)
        Returns:
            hidden_states: (B, L, hidden_dim) GRU outputs at each position
        """
        seq_emb = self.item_emb(input_seq)
        seq_emb = self.emb_dropout(seq_emb)

        gru_out, _ = self.gru(seq_emb)

        gru_out = self.output_dropout(gru_out)
        return gru_out

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for all items from the last hidden state.

        Args:
            input_seq: (B, L) padded item ID sequence
        Returns:
            scores: (B, num_items+1) raw scores for each item
        """
        seq_emb = self.item_emb(input_seq)
        seq_emb = self.emb_dropout(seq_emb)
        _, hidden = self.gru(seq_emb)
        last_hidden = hidden[-1]
        return self._project_hidden(last_hidden)

    def forward_from_emb(self, seq_emb: torch.Tensor, input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward from pre-computed embeddings (for gradient computation).
        Args:
            seq_emb: (B, L, emb_dim) item embeddings
        Returns:
            scores: (B, num_items+1) raw scores for each item
        """
        _, hidden = self.gru(seq_emb)
        last_hidden = hidden[-1]
        return self._project_hidden(last_hidden)

    def compute_loss(
        self,
        input_seq: torch.Tensor,
        target_pos: torch.Tensor,
        target_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        CE (full softmax) loss at ALL positions (same as SASRec).

        When target_pos is (B, L), compute per-position CE loss.
        When target_pos is (B,), compute last-position CE loss.

        Args:
            input_seq: (B, L) input sequence
            target_pos: (B, L) per-position targets OR (B,) last-item target
            target_neg: ignored
        Returns:
            loss: scalar CE loss
        """
        if target_pos.dim() == 1:
            # Single target per user (backward compat)
            scores = self.predict(input_seq)  # (B, num_items+1)
            loss = F.cross_entropy(scores, target_pos)
            return loss

        # Per-position training (like SASRec)
        hidden_states = self.forward(input_seq)  # (B, L, hidden_dim)
        emb_out = self.dense(hidden_states)  # (B, L, emb_dim)
        logits = emb_out @ self.item_emb.weight.T  # (B, L, num_items+1)

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
