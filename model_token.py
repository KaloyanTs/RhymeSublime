import torch
import re


class TokenLSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for (a, s) in source)

        sents = []
        for (a, seq) in source:
            enc = []
            for w in seq:
                if w in self.word2ind:
                    enc.append(self.word2ind[w])
                elif self._letter_re.match(w):
                    for t in self._greedy_encode_word(w):
                        enc.append(self.word2ind.get(t, self.unkTokenIdx))
                else:
                    enc.append(self.unkTokenIdx)
            sents.append(enc)
        auths = [self.auth2id.get(a, 0) for (a, s) in source]

        sents_padded = [s + (m - len(s)) * [self.padTokenIdx] for s in sents]
        X = torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))  # (T,B)
        auth = torch.tensor(auths, dtype=torch.long, device=device)               # (B,)
        return X, auth

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)
        try:
            print("[TokenLSTM] Saved:", fileName)
        except Exception:
            pass

    def load(self, fileName, device=torch.device("cuda:0")):
        self.load_state_dict(torch.load(fileName, map_location=device))
        try:
            print("[TokenLSTM] Loaded:", fileName)
        except Exception:
            pass

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
        lambda_rhyme=1.0,
    ):
        super().__init__()

        self.word2ind = word2ind
        self.auth2id = auth2id
        self.lstm_layers = lstm_layers

        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.spaceTokenIdx = word2ind.get(" ", None)
        self.lineEndTokenIdx = word2ind.get("\n", None)

        self.lambda_rhyme = float(lambda_rhyme)

        self.last_lm_loss = None
        self.last_rhyme_loss = None

        self.lstm = torch.nn.LSTM(embed_size, hidden_size, lstm_layers, dropout=dropout)
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.embed_auth_cell = torch.nn.Embedding(len(auth2id), hidden_size)
        self.embed_auth_out = torch.nn.Embedding(len(auth2id), hidden_size)
        self.projection = torch.nn.Linear(hidden_size, len(word2ind))

        self.id2tok = [None] * len(word2ind)
        for tok, i in word2ind.items():
            if 0 <= i < len(self.id2tok):
                self.id2tok[i] = tok

        self._letter_re = re.compile(r"^[A-Za-z\u0400-\u04FF\u0500-\u052F]+$")
        self._greedy_vocab = [t for t in word2ind.keys() if self._letter_re.match(t)]
        self._greedy_vocab.sort(key=len, reverse=True)

        try:
            print(
                "[TokenLSTM] Init:",
                f"vocab={len(word2ind)}",
                f"embed={embed_size}",
                f"layers={lstm_layers}",
                f"dropout={dropout}",
                f"lambda_rhyme={self.lambda_rhyme}",
            )
        except Exception:
            pass

    def _greedy_encode_word(self, word: str):
        if not word:
            return []
        i = 0
        out = []
        L = len(word)
        while i < L:
            matched = False
            for tok in self._greedy_vocab:
                tl = len(tok)
                if tl == 0 or i + tl > L:
                    continue
                if word[i:i+tl] == tok:
                    out.append(tok)
                    i += tl
                    matched = True
                    break
            if not matched:
                out.append(word[i])
                i += 1
        return out

    def _rhyme_loss_last(self, logits_seq, y, source_lengths):
        if self.lambda_rhyme <= 0.0 or self.lineEndTokenIdx is None:
            return logits_seq.new_tensor(0.0)

        _, B, _ = logits_seq.shape

        pos_t_list, pos_b_list, tgt_list = [], [], []

        for b in range(B):
            L = int(source_lengths[b])
            if L <= 0:
                continue

            yb = y[:L, b]
            nl_pos = (yb == self.lineEndTokenIdx).nonzero(as_tuple=False).flatten()
            if nl_pos.numel() < 2:
                continue

            nl_pos_list = nl_pos.tolist()
            for i in range(1, len(nl_pos_list)):
                t_prev = nl_pos_list[i - 1]
                t_cur = nl_pos_list[i]
                if t_prev - 1 < 0 or t_cur - 1 < 0:
                    continue

                prev_last_tok = yb[t_prev - 1]
                pos_t_list.append(torch.tensor([t_cur - 1], device=logits_seq.device))
                pos_b_list.append(torch.tensor([b], device=logits_seq.device))
                tgt_list.append(prev_last_tok.view(1))

        if not tgt_list:
            return logits_seq.new_tensor(0.0)

        pos_t = torch.cat(pos_t_list)
        pos_b = torch.cat(pos_b_list)
        tgt = torch.cat(tgt_list)

        sel_logits = logits_seq[pos_t, pos_b, :]
        ce = torch.nn.functional.cross_entropy(sel_logits, tgt, reduction="mean")

        return self.lambda_rhyme * ce

    def forward(self, source):
        X, auth = self.preparePaddedBatch(source)   # X: (T,B)
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

        rhyme_loss = self._rhyme_loss_last(logits_seq, y, source_lengths)

        self.last_lm_loss = lm_loss
        self.last_rhyme_loss = rhyme_loss

        return lm_loss + rhyme_loss
