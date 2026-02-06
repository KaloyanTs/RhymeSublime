from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharAuthLSTM(nn.Module):
    """
    Char-level LM with author-conditioned initial hidden/cell state.

    IMPORTANT:
      - "Right-to-left" poems are achieved ONLY via 1rocessing (reverse each line's characters).
      - Standard training: next-char cross-entropy (no rhyme loss).

    Compatibility with your generator:
      - attributes: embed, lstm, projection, auth2id, unkTokenIdx, lineEndTokenIdx,
                    embed_auth_out, embed_auth_cell, lstm_layers
    """

    def __init__(
        self,
        vocab_size: int,
        auth2id: Dict[str, int],
        *,
        emb_dim: int = 128,
        hidden_dim: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        unk_token_idx: int = 0,
        line_end_token_idx: Optional[int] = None,
        tie_weights: bool = False,
        loss_ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be > 0")
        if emb_dim <= 0 or hidden_dim <= 0:
            raise ValueError("emb_dim and hidden_dim must be > 0")
        if not auth2id:
            raise ValueError("auth2id must be a non-empty dict")

        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.lstm_layers = int(lstm_layers)
        self.dropout = float(dropout)
        self.loss_ignore_index = int(loss_ignore_index)

        # Keep these names for compatibility with existing code.
        self.unkTokenIdx = int(unk_token_idx)
        self.lineEndTokenIdx = int(line_end_token_idx) if line_end_token_idx is not None else None
        self.auth2id = dict(auth2id)

        self.num_authors = int(max(self.auth2id.values()) + 1)

        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)

        # Author-conditioned init state (replicated across layers).
        self.embed_auth_out = nn.Embedding(self.num_authors, self.hidden_dim)
        self.embed_auth_cell = nn.Embedding(self.num_authors, self.hidden_dim)

        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=False,  # matches your generator
        )

        self.projection = nn.Linear(self.hidden_dim, self.vocab_size, bias=True)

        self.tie_weights = bool(tie_weights)
        if self.tie_weights:
            if self.emb_dim != self.hidden_dim:
                raise ValueError("tie_weights=True requires emb_dim == hidden_dim")
            self.projection.weight = self.embed.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_auth_out.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_auth_cell.weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.projection.bias)
        if not self.tie_weights:
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def flatten_lstm_parameters(self) -> None:
        """Call after moving to CUDA for a small cuDNN speed bump."""
        self.lstm.flatten_parameters()

    def init_state(self, author_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        author_ids:
          - scalar -> returns (L, H) unbatched
          - (B,)   -> returns (L, B, H) batched
        """
        if author_ids.dtype != torch.long:
            raise TypeError("author_ids must be torch.long")

        if author_ids.dim() == 0:
            h0 = self.embed_auth_out(author_ids)   # (H,)
            c0 = self.embed_auth_cell(author_ids)  # (H,)
            h0 = h0.unsqueeze(0).expand(self.lstm_layers, -1).contiguous()  # (L,H)
            c0 = c0.unsqueeze(0).expand(self.lstm_layers, -1).contiguous()  # (L,H)
            return h0, c0

        if author_ids.dim() == 1:
            h0 = self.embed_auth_out(author_ids)   # (B,H)
            c0 = self.embed_auth_cell(author_ids)  # (B,H)
            h0 = h0.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()  # (L,B,H)
            c0 = c0.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()  # (L,B,H)
            return h0, c0

        raise ValueError("author_ids must be a scalar or shape (B,)")

    def _compute_logits(
        self,
        input_ids: torch.LongTensor,
        author_ids: torch.LongTensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long")

        if input_ids.dim() == 1:
            # (T,)
            if hx is None:
                hx = self.init_state(author_ids)  # (L,H)
            x = self.embed(input_ids)            # (T,E)
            o, (hT, cT) = self.lstm(x, hx=hx)    # o: (T,H)
            logits = self.projection(o)          # (T,V)
            return logits, (hT, cT)

        if input_ids.dim() == 2:
            # (T,B)
            if hx is None:
                hx = self.init_state(author_ids)  # (L,B,H)
            x = self.embed(input_ids)             # (T,B,E)
            o, (hT, cT) = self.lstm(x, hx=hx)     # o: (T,B,H)
            logits = self.projection(o)           # (T,B,V)
            return logits, (hT, cT)

        raise ValueError("input_ids must be shape (T,) or (T,B)")

    def forward(
        self,
        input_ids: torch.LongTensor,
        author_ids: torch.LongTensor,
        *,
        target_ids: Optional[torch.LongTensor] = None,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_logits: bool = False,
        return_state: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ]:
        logits, (hT, cT) = self._compute_logits(input_ids, author_ids, hx)

        # Build targets / align shapes for CE.
        if target_ids is None:
            if input_ids.size(0) < 2:
                raise ValueError("Need T>=2 to compute shifted LM loss when target_ids is None")

            if input_ids.dim() == 1:
                # logits: (T,V) -> use first T-1
                logits_for_loss = logits[:-1, :]         # (T-1,V)
                targets = input_ids[1:]                  # (T-1,)
                loss = F.cross_entropy(
                    logits_for_loss, targets, ignore_index=self.loss_ignore_index
                )
            else:
                # logits: (T,B,V) -> use first T-1
                logits_for_loss = logits[:-1, :, :]      # (T-1,B,V)
                targets = input_ids[1:, :]               # (T-1,B)
                loss = F.cross_entropy(
                    logits_for_loss.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    ignore_index=self.loss_ignore_index,
                )
        else:
            if target_ids.dtype != torch.long:
                raise TypeError("target_ids must be torch.long")

            # If target_ids matches input_ids length, align with logits length.
            # You can also pass already-aligned target_ids (same leading dims as logits).
            if input_ids.dim() == 1:
                # logits: (T,V)
                if target_ids.dim() != 1:
                    raise ValueError("target_ids must be (T,) for unbatched input")
                if target_ids.size(0) == logits.size(0):
                    loss = F.cross_entropy(logits, target_ids, ignore_index=self.loss_ignore_index)
                else:
                    # assume already aligned (e.g., (T-1,))
                    if target_ids.size(0) != logits.size(0):
                        # allow logits trimmed externally; safest is to require equality here
                        raise ValueError("target_ids length must match logits length for loss")
                    loss = F.cross_entropy(logits, target_ids, ignore_index=self.loss_ignore_index)
            else:
                # logits: (T,B,V)
                if target_ids.dim() != 2:
                    raise ValueError("target_ids must be (T,B) for batched input")
                if target_ids.size(0) != logits.size(0) or target_ids.size(1) != logits.size(1):
                    raise ValueError("target_ids must match logits (T,B) for loss")
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    target_ids.reshape(-1),
                    ignore_index=self.loss_ignore_index,
                )

        if not return_logits and not return_state:
            return loss
        if return_logits and not return_state:
            return loss, logits
        # return_state=True
        return loss, logits if return_logits else loss, (hT, cT)
