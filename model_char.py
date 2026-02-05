import torch
from functools import lru_cache
from stress import predict as stress_predict


# ---- Stress predictor cache (BIG speedup) ----
@lru_cache(maxsize=200_000)
def cached_stress_idx(word: str) -> int:
    """
    Returns 0-based stress char index, or -1 if stress can't be predicted.
    Cached across calls to avoid repeated model inference.
    """
    try:
        return int(stress_predict(word))
    except Exception:
        return -1


class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for (a, s) in source)

        sents = [[self.word2ind.get(w, self.unkTokenIdx) for w in s] for (a, s) in source]
        auths = [self.auth2id.get(a, 0) for (a, s) in source]

        sents_padded = [s + (m - len(s)) * [self.padTokenIdx] for s in sents]
        X = torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))  # (T,B)
        auth = torch.tensor(auths, dtype=torch.long, device=device)               # (B,)
        return X, auth

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName, device=torch.device("cuda:0")):
        self.load_state_dict(torch.load(fileName, map_location=device))

    def __init__(
        self,
        embed_size,
        hidden_size,
        auth2id,
        word2ind,
        unkToken,
        padToken,
        endToken,
        lstm_layers,
        dropout,
        # rhyme knobs (AAAA)
        k_rhyme=3,            # set >0 to enable, e.g. 3
        lambda_rhyme=1.0,     # set >0 to enable, e.g. 0.2
    ):
        super(LSTMLanguageModelPack, self).__init__()

        self.word2ind = word2ind
        self.auth2id = auth2id
        self.lstm_layers = lstm_layers

        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.spaceTokenIdx = word2ind.get(" ", None)
        self.lineEndTokenIdx = word2ind.get("\n", None)

        self.k_rhyme = int(k_rhyme)
        self.lambda_rhyme = float(lambda_rhyme)

        self.last_lm_loss = None
        self.last_rhyme_loss = None

        self.lstm = torch.nn.LSTM(embed_size, hidden_size, lstm_layers, dropout=dropout)
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.embed_auth_cell = torch.nn.Embedding(len(auth2id), hidden_size)
        self.embed_auth_out = torch.nn.Embedding(len(auth2id), hidden_size)
        self.projection = torch.nn.Linear(hidden_size, len(word2ind))

        # id -> char mapping for reconstructing words
        self.id2char = [None] * len(word2ind)
        for ch, i in word2ind.items():
            if 0 <= i < len(self.id2char):
                self.id2char[i] = ch

    def _rhyme_loss_aaaa(self, logits_seq, y, source_lengths):
        """
        logits_seq: (T,B,V) for predicting y (which is X[1:])
        y:         (T,B)
        source_lengths: list[int] true lengths in y per batch element (no pads), i.e. len(s)-1
        """
        if self.lambda_rhyme <= 0.0:
            return logits_seq.new_tensor(0.0)

        _, B, _ = logits_seq.shape

        pos_t_chunks = []
        pos_b_chunks = []
        tgt_chunks = []

        for b in range(B):
            L = int(source_lengths[b])
            if L <= 0:
                continue

            yb = y[:L, b]  # valid part only
            if self.lineEndTokenIdx is None:
                continue

            nl_pos = (yb == self.lineEndTokenIdx).nonzero(as_tuple=False).flatten()
            if nl_pos.numel() < 2:
                continue
            nl_pos_list = nl_pos.tolist()

            # iterate over line endings, starting from 2nd newline
            for i in range(1, len(nl_pos_list)):
                t_prev = nl_pos_list[i - 1]  # previous line end index in yb
                t_cur = nl_pos_list[i]       # current line end index in yb

                prev_line_start = (nl_pos_list[i - 2] + 1) if i >= 2 else 0

                # Find start of previous line's LAST word (vectorized: last space)
                start_prev_word = prev_line_start
                if self.spaceTokenIdx is not None and t_prev > prev_line_start:
                    seg = yb[prev_line_start:t_prev]
                    space_pos = (seg == int(self.spaceTokenIdx)).nonzero(as_tuple=False).flatten()
                    if space_pos.numel() > 0:
                        start_prev_word = prev_line_start + int(space_pos[-1].item()) + 1

                if start_prev_word >= t_prev:
                    continue

                prev_word_ids = yb[start_prev_word:t_prev].tolist()
                # reconstruct word string for stress prediction
                chars = []
                for iw in prev_word_ids:
                    ch = self.id2char[iw] if 0 <= iw < len(self.id2char) else None
                    if ch is not None:
                        chars.append(ch)
                prev_word = "".join(chars)
                if not prev_word:
                    continue

                # cached stress prediction (huge speedup)
                stress_idx = cached_stress_idx(prev_word)
                if stress_idx < 0 or stress_idx >= len(prev_word):
                    continue

                # full suffix length from stressed char to end of word
                full_k = len(prev_word) - stress_idx
                if full_k <= 0:
                    continue

                # cap suffix length with k_rhyme (if >0)
                dynamic_k = min(full_k, self.k_rhyme) if self.k_rhyme > 0 else full_k
                if dynamic_k <= 0:
                    continue

                # Ensure current line has enough chars before newline to enforce rhyme
                cur_line_start = t_prev + 1
                cur_len = t_cur - cur_line_start
                if cur_len < dynamic_k:
                    continue

                # Map stress-based suffix to token indices in yb:
                # prev word spans [t_prev - len(prev_word), t_prev)
                prev_suffix_start = (t_prev - len(prev_word)) + stress_idx
                if prev_suffix_start < 0 or prev_suffix_start + dynamic_k > t_prev:
                    continue

                prev_suffix = yb[prev_suffix_start:t_prev]  # length full_k

                # Vectorized: select all positions at once (removes inner j-loop)
                t_idx = torch.arange(
                    t_cur - dynamic_k, t_cur,
                    device=logits_seq.device,
                    dtype=torch.long,
                )
                b_idx = torch.full(
                    (dynamic_k,),
                    b,
                    device=logits_seq.device,
                    dtype=torch.long,
                )
                tgt_tok = prev_suffix[:dynamic_k]  # tensor

                # guard (should already hold) to avoid out-of-range
                if t_idx.min().item() < 0 or t_idx.max().item() >= L:
                    continue

                pos_t_chunks.append(t_idx)
                pos_b_chunks.append(b_idx)
                tgt_chunks.append(tgt_tok)

        if not tgt_chunks:
            return logits_seq.new_tensor(0.0)

        pos_t = torch.cat(pos_t_chunks, dim=0)
        pos_b = torch.cat(pos_b_chunks, dim=0)
        tgt = torch.cat(tgt_chunks, dim=0)

        sel_logits = logits_seq[pos_t, pos_b, :]  # (M,V)
        ce = torch.nn.functional.cross_entropy(sel_logits, tgt, reduction="mean")
        return self.lambda_rhyme * ce

    def forward(self, source):
        X, auth = self.preparePaddedBatch(source)   # X: (T,B)
        # Predict X[1:] from X[:-1]
        E = self.embed(X[:-1])                      # (T-1,B,E)

        h0 = self.embed_auth_out(auth).unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        c0 = self.embed_auth_cell(auth).unsqueeze(0).repeat(self.lstm_layers, 1, 1)

        source_lengths = [len(s) - 1 for (a, s) in source]  # lengths in X[1:] (no pads)

        outputPacked, _ = self.lstm(
            torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted=False),
            hx=(h0, c0),
        )
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)  # (T-1,B,H)

        logits_seq = self.projection(output)        # (T-1,B,V)
        y = X[1:]                                   # (T-1,B)

        lm_loss = torch.nn.functional.cross_entropy(
            logits_seq.flatten(0, 1),
            y.flatten(0, 1),
            ignore_index=self.padTokenIdx,
        )

        rhyme_loss = self._rhyme_loss_aaaa(logits_seq, y, source_lengths)

        self.last_lm_loss = lm_loss
        self.last_rhyme_loss = rhyme_loss

        return lm_loss + rhyme_loss
