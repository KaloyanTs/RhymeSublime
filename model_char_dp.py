import math
import re
import torch
import torch.nn.functional as F
import csv
from stress import predict as stress_predict

# ---------------------------------------------------------------------
# Phonetic feature distance (same as generation code, but used for DP loss)
# ---------------------------------------------------------------------

VOWELS_BG = set(list("аеиоуъюяѝАЕИОУЪЮЯЍ"))

PHONETIC_FEATURES = {}
for ch, h, b in [
    ("а", 3, 3), ("е", 2, 1), ("и", 1, 1), ("о", 2, 3),
    ("у", 1, 3), ("ъ", 2, 2), ("я", 3, 1), ("ю", 1, 3),
]:
    PHONETIC_FEATURES[ch] = [1, 1, 0, 0, h, b]

for ch, v, p, m in [
    ("б", 1, 1, 1), ("п", 0, 1, 1),
    ("в", 1, 2, 2), ("ф", 0, 2, 2),
    ("д", 1, 3, 1), ("т", 0, 3, 1),
    ("з", 1, 3, 2), ("с", 0, 3, 2),
    ("ж", 1, 4, 2), ("ш", 0, 4, 2),
    ("г", 1, 5, 1), ("к", 0, 5, 1),
    ("м", 1, 1, 4), ("н", 1, 3, 4),
    ("л", 1, 3, 5), ("р", 1, 3, 5),
    ("й", 1, 4, 5),
    ("х", 0, 5, 2),
    ("ц", 0, 3, 3), ("ч", 0, 4, 3),
    ("щ", 0, 6, 2), ("ь", 0, 0, 0),
]:
    PHONETIC_FEATURES[ch] = [0, v, p, m, 0, 0]


# ---------------------------
# Stress lookup and prediction
# ---------------------------

_STRESS_MAP = None


def _load_stress_map(path: str = "bg_dict_csv/single_stress.csv"):
    mp = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().lower() == "id":
                    continue
                if len(row) >= 3:
                    name = row[1].strip()
                    stressed = row[2].strip()
                    if name:
                        mp[name.lower()] = stressed
    except Exception:
        mp = {}
    print(f"Loaded stress map with {len(mp)} entries from {path}")
    return mp


def _stress_index_from_stressed(stressed: str, base: str) -> int:
    i = stressed.find("`")
    if i <= 0:
        return 0
    sidx = i - 1
    if sidx < 0:
        sidx = 0
    if sidx >= len(base):
        sidx = max(0, len(base) - 1)
    return sidx


def _get_stress_index_word(word: str) -> int:
    global _STRESS_MAP
    if _STRESS_MAP is None:
        _STRESS_MAP = _load_stress_map()
    stressed = _STRESS_MAP.get(word.lower())
    if isinstance(stressed, str) and stressed:
        return _stress_index_from_stressed(stressed, word)
    try:
        return int(stress_predict(word))
    except Exception:
        for j in range(len(word) - 1, -1, -1):
            if word[j] in VOWELS_BG:
                return j
        return 0


def phonetic_dist(a: str, b: str) -> float:
    """
    Distance between Bulgarian letters based on a small handcrafted feature vector.
    Unknowns default to 1.0 (same behavior as your original code).
    """
    if a == b:
        return 0.0
    a_l = a.lower()
    b_l = b.lower()
    fa = PHONETIC_FEATURES.get(a_l)
    fb = PHONETIC_FEATURES.get(b_l)
    if fa is None or fb is None:
        return 1.0
    return math.sqrt(sum((fa[i] - fb[i]) ** 2 for i in range(len(fa))))


def build_phon_cost_matrix(id2tok):
    """
    Build C[v_pred, v_tgt] = phonetic_dist(tok_pred, tok_tgt).
    Safe for non-letter/special tokens (falls back to 1.0).
    """
    V = len(id2tok)
    C = torch.empty((V, V), dtype=torch.float32)
    for i in range(V):
        a = id2tok[i] if id2tok[i] is not None else ""
        for j in range(V):
            b = id2tok[j] if id2tok[j] is not None else ""
            C[i, j] = float(phonetic_dist(a, b))
    return C


# ---------------------------------------------------------------------
# Differentiable DP: soft edit-distance with expected phonetic sub cost
# ---------------------------------------------------------------------

def softmin3(a, b, c, gamma: float):
    """
    softmin(a,b,c) = -gamma * log(exp(-a/g)+exp(-b/g)+exp(-c/g))
    """
    x = torch.stack([a, b, c], dim=0)
    return -gamma * torch.logsumexp(-x / gamma, dim=0)


def rhyme_dp_loss_from_logits(
    tail_logits: torch.Tensor,     # [T, V]
    target_idx: torch.Tensor,      # [M]
    phon_cost: torch.Tensor,       # [V, V]
    ins_del_cost: float = 10.0,
    gamma: float = 1.0,
    first_char_mismatch_cost: float = 10.0,
):
    """
    Differentiable version of rhyme_penalty_str(base_tail, cand_tail), but:
      - cand_tail is represented by a *sequence of logits* (teacher-forced positions)
      - substitution cost is expected phonetic distance under p(char|context)
      - min is replaced by softmin (entropic smoothing)

    Returns a scalar tensor (0-dim).
    """
    T, V = tail_logits.shape
    M = int(target_idx.numel())

    if T <= 0 or M <= 0:
        return tail_logits.new_zeros(())

    # probs over predicted chars
    p = F.softmax(tail_logits, dim=-1)  # [T, V]

    # substitution costs: sub[t, j] = sum_v p[t,v] * C[v, target[j]]
    C_cols = phon_cost[:, target_idx]        # [V, M]
    sub = p @ C_cols                         # [T, M]

    # DP table (soft edit distance)
    dp = tail_logits.new_empty((T + 1, M + 1))
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        dp[i, 0] = dp[i - 1, 0] + ins_del_cost
    for j in range(1, M + 1):
        dp[0, j] = dp[0, j - 1] + ins_del_cost

    for i in range(1, T + 1):
        for j in range(1, M + 1):
            dp[i, j] = softmin3(
                dp[i - 1, j] + ins_del_cost,          # deletion
                dp[i, j - 1] + ins_del_cost,          # insertion
                dp[i - 1, j - 1] + sub[i - 1, j - 1], # substitution
                gamma=gamma,
            )

    # Differentiable "first char mismatch + 10":
    # penalty = 10 * (1 - P(pred_first == target_first))
    first_tgt = int(target_idx[0].item())
    first_match_prob = p[0, first_tgt]
    return dp[T, M] + first_char_mismatch_cost * (1.0 - first_match_prob)


# ---------------------------------------------------------------------
# Char LSTM LM + differentiable DP rhyme loss
# ---------------------------------------------------------------------

def _is_cyrillic_char(ch: str) -> bool:
    return ('\u0400' <= ch <= '\u04FF') or ('\u0500' <= ch <= '\u052F')


class CharLSTMLanguageModelPack(torch.nn.Module):
    """
    Same structure as TokenLSTMLanguageModelPack, but assumes the input sequence
    is already a list of *char tokens*.

    Total loss = LM cross-entropy + lambda_rhyme * differentiable DP rhyme loss.
    """

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device

        sents = []
        for (a, seq) in source:
            seq_list = list(seq) if isinstance(seq, str) else seq
            enc = [self.word2ind.get(t, self.unkTokenIdx) for t in seq_list]
            sents.append(enc)

        m = max(len(s) for s in sents) if sents else 1
        auths = [self.auth2id.get(a, 0) for (a, _s) in source]
        sents_padded = [s + (m - len(s)) * [self.padTokenIdx] for s in sents]

        X = torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))  # (T,B)
        auth = torch.tensor(auths, dtype=torch.long, device=device)               # (B,)
        return X, auth

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)
        try:
            print("[CharLSTM] Saved:", fileName)
        except Exception:
            pass

    def load(self, fileName, device=torch.device("cuda:0")):
        self.load_state_dict(torch.load(fileName, map_location=device))
        try:
            print("[CharLSTM] Loaded:", fileName)
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
        dp_gamma=1.0,
        dp_ins_del_cost=10.0,
        dp_first_char_cost=10.0,
        dp_tail_max=16,
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

        # Weight of the DP rhyme loss (kept name for compatibility with old training scripts)
        self.lambda_rhyme = float(lambda_rhyme)

        # DP hyperparams
        self.dp_gamma = float(dp_gamma)
        self.dp_ins_del_cost = float(dp_ins_del_cost)
        self.dp_first_char_cost = float(dp_first_char_cost)
        self.dp_tail_max = int(dp_tail_max) if dp_tail_max is not None else 0

        self.last_lm_loss = None
        self.last_rhyme_loss = None

        self.lstm = torch.nn.LSTM(embed_size, hidden_size, lstm_layers, dropout=dropout)
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.embed_auth_cell = torch.nn.Embedding(len(auth2id), hidden_size)
        self.embed_auth_out = torch.nn.Embedding(len(auth2id), hidden_size)
        self.projection = torch.nn.Linear(hidden_size, len(word2ind))

        # id -> token mapping
        self.id2tok = [None] * len(word2ind)
        for tok, i in word2ind.items():
            if 0 <= i < len(self.id2tok):
                self.id2tok[i] = tok

        # Phonetic cost matrix as a buffer (moves with .to(device))
        C = build_phon_cost_matrix(self.id2tok)
        self.register_buffer("phon_cost", C)

        try:
            print(
                "[CharLSTM] Init:",
                f"vocab={len(word2ind)}",
                f"embed={embed_size}",
                f"layers={lstm_layers}",
                f"dropout={dropout}",
                f"lambda_rhyme={self.lambda_rhyme}",
                f"dp_gamma={self.dp_gamma}",
                f"dp_tail_max={self.dp_tail_max}",
            )
        except Exception:
            pass

    def _is_cyrillic_tok(self, idx: int) -> bool:
        tok = self.id2tok[int(idx)] if 0 <= int(idx) < len(self.id2tok) else ""
        return isinstance(tok, str) and len(tok) == 1 and _is_cyrillic_char(tok)

    def _is_vowel_tok(self, idx: int) -> bool:
        tok = self.id2tok[int(idx)] if 0 <= int(idx) < len(self.id2tok) else ""
        return tok in VOWELS_BG

    def _extract_tail_span(self, yb: torch.Tensor, start: int, end_excl: int):
        """
        Extract rhyme tail from the last word in yb[start:end_excl] (excluding newline).
        Tail starts at the last vowel in the last word (fallback: whole last word).

        Returns:
          tail_ids (1D LongTensor, on same device as yb),
          tail_start_idx (int, index into yb),
          tail_end_idx (int, index into yb)
        If no valid tail found, returns (None, None, None).
        """
        if end_excl <= start:
            return None, None, None

        # Find last Cyrillic letter in the line
        i = end_excl - 1
        while i >= start and not self._is_cyrillic_tok(int(yb[i].item())):
            i -= 1
        if i < start:
            return None, None, None

        tail_end = i

        # Find start of last word (contiguous Cyrillic letters)
        j = tail_end
        while j >= start and self._is_cyrillic_tok(int(yb[j].item())):
            j -= 1
        word_start = j + 1
        word_ids = yb[word_start:tail_end + 1]  # includes only Cyrillic letters

        if word_ids.numel() <= 0:
            return None, None, None

        # Stress index: prefer dictionary/predictor, fallback to last vowel
        word = ""
        for t in word_ids:
            idx = int(t.item())
            tok = self.id2tok[idx] if 0 <= idx < len(self.id2tok) else ""
            if isinstance(tok, str):
                word_part = tok
            else:
                word_part = ""
            word = word + word_part
        sidx = _get_stress_index_word(word)
        if sidx < 0:
            sidx = 0
        if sidx >= int(word_ids.numel()):
            sidx = max(0, int(word_ids.numel()) - 1)

        tail_ids = word_ids[sidx:].clone()
        tail_start = word_start + sidx

        # Optional cap on tail length for speed
        if self.dp_tail_max and tail_ids.numel() > self.dp_tail_max:
            tail_ids = tail_ids[-self.dp_tail_max:].clone()
            tail_start = tail_end - int(tail_ids.numel()) + 1

        return tail_ids, int(tail_start), int(tail_end)

    def _rhyme_loss_dp(self, logits_seq, y, source_lengths):
        """
        Differentiable DP rhyme loss:
        For each pair of consecutive lines (separated by '\n'), encourage the current line's
        tail to match the previous line's tail, using soft edit distance with phonetic substitution cost.
        """
        if self.lambda_rhyme <= 0.0 or self.lineEndTokenIdx is None:
            return logits_seq.new_tensor(0.0)

        _, B, _ = logits_seq.shape
        losses = []

        for b in range(B):
            L = int(source_lengths[b])
            if L <= 0:
                continue

            yb = y[:L, b]
            logits_b = logits_seq[:L, b, :]

            nl_pos = (yb == self.lineEndTokenIdx).nonzero(as_tuple=False).flatten()
            if nl_pos.numel() < 2:
                continue

            # Build line spans: [start, end_excl) for each line (excluding newline)
            nl_list = nl_pos.tolist()
            spans = []
            s = 0
            for t_nl in nl_list:
                if t_nl >= s:
                    spans.append((s, t_nl))
                s = t_nl + 1

            if len(spans) < 2:
                continue

            # Compare each line with the previous one (couplet-style)
            for k in range(1, len(spans)):
                prev_start, prev_end = spans[k - 1]
                cur_start, cur_end = spans[k]

                base_tail_ids, _, _ = self._extract_tail_span(yb, prev_start, prev_end)
                cand_tail_ids, cand_t_start, cand_t_end = self._extract_tail_span(yb, cur_start, cur_end)

                if base_tail_ids is None or cand_tail_ids is None:
                    continue
                if base_tail_ids.numel() <= 0 or cand_tail_ids.numel() <= 0:
                    continue

                # logits for the candidate tail positions
                tail_logits = logits_b[cand_t_start:cand_t_end + 1, :]  # [T, V]
                target_idx = base_tail_ids.to(dtype=torch.long)

                dp_val = rhyme_dp_loss_from_logits(
                    tail_logits=tail_logits,
                    target_idx=target_idx,
                    phon_cost=self.phon_cost,  # buffer
                    ins_del_cost=self.dp_ins_del_cost,
                    gamma=self.dp_gamma,
                    first_char_mismatch_cost=self.dp_first_char_cost,
                )
                losses.append(dp_val)

        if not losses:
            return logits_seq.new_tensor(0.0)

        return self.lambda_rhyme * torch.stack(losses).mean()

    def forward(self, source):
        X, auth = self.preparePaddedBatch(source)  # X: (T,B)

        E = self.embed(X[:-1])  # (T-1,B,E)

        h0 = self.embed_auth_out(auth).unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        c0 = self.embed_auth_cell(auth).unsqueeze(0).repeat(self.lstm_layers, 1, 1)

        source_lengths = [len(s) - 1 for (a, s) in source]  # lengths in X[1:]

        outputPacked, _ = self.lstm(
            torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted=False),
            hx=(h0, c0),
        )
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)  # (T-1,B,H)

        logits_seq = self.projection(output)  # (T-1,B,V)
        y = X[1:]                             # (T-1,B)

        lm_loss = torch.nn.functional.cross_entropy(
            logits_seq.flatten(0, 1),
            y.flatten(0, 1),
            ignore_index=self.padTokenIdx,
        )

        rhyme_loss = self._rhyme_loss_dp(logits_seq, y, source_lengths)

        self.last_lm_loss = lm_loss
        self.last_rhyme_loss = rhyme_loss

        return lm_loss + rhyme_loss
