import numpy as np
import torch
import torch.nn.functional as F
import math
import csv
from stress import predict as stress_predict

# ---------------------------------------------------------------------
# Same phonetic + hard DP utilities as before (for debugging/evaluation).
# Training uses differentiable DP inside the model.
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

def phonetic_dist(a: str, b: str):
    if a == b:
        return 0.0
    a_l = a.lower()
    b_l = b.lower()
    fa = PHONETIC_FEATURES.get(a_l)
    fb = PHONETIC_FEATURES.get(b_l)
    if fa is None or fb is None:
        return 1.0
    return math.sqrt(sum((fa[i] - fb[i]) ** 2 for i in range(len(fa))))


def rhyme_penalty_str(base_tail: str, cand_tail: str):
    """
    Hard min edit distance (non-differentiable). Used only for debug scoring in generation.
    """
    if not base_tail or not cand_tail:
        return 0.0
    n, m = len(base_tail), len(cand_tail)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + 10.0
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + 10.0
    for i in range(1, n + 1):
        ai = base_tail[i - 1]
        for j in range(1, m + 1):
            bj = cand_tail[j - 1]
            dp[i][j] = min(
                dp[i - 1][j] + 10.0,
                dp[i][j - 1] + 10.0,
                dp[i - 1][j - 1] + phonetic_dist(ai, bj),
            )
    dist = dp[n][m]
    if base_tail and cand_tail and base_tail[0].lower() != cand_tail[0].lower():
        dist += 10.0
    return dist


def repetition_penalty(line_ids, w_rep, id2tok):
    if w_rep <= 0.0:
        return 0.0
    line_text = "".join(id2tok[t] for t in line_ids)
    pen = 0.0
    run = 1
    for i in range(1, len(line_text)):
        if line_text[i] == line_text[i - 1]:
            run += 1
            if run >= 2:
                ch = line_text[i]
                pen += 0.5 if ch in VOWELS_BG else 1.0
        else:
            run = 1
    return pen


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
    return mp


def _stress_index_from_stressed(stressed: str, base: str) -> int:
    i = stressed.find("`")
    if i <= 0:
        return 0
    sidx = i - 1  # backtick is after the stressed character
    if sidx < 0:
        sidx = 0
    if sidx >= len(base):
        sidx = max(0, len(base) - 1)
    return sidx


def get_stress_index(word: str) -> int:
    global _STRESS_MAP
    if _STRESS_MAP is None:
        _STRESS_MAP = _load_stress_map()
    stressed = _STRESS_MAP.get(word.lower())
    if isinstance(stressed, str) and stressed:
        idx = _stress_index_from_stressed(stressed, word)
        # print(f"[Stress] Word '{word}': found in map, stressed='{stressed}', idx={idx}")
        return idx
    try:
        idx = int(stress_predict(word))
        #print(f"[Stress] Word '{word}': predicted stress idx={idx}")
        return idx
    except Exception as e:
        # print(f"[Stress] Word '{word}': stress_predict failed ({e}), searching for vowel")
        for j in range(len(word) - 1, -1, -1):
            if word[j] in VOWELS_BG:
                # print(f"[Stress] Word '{word}': using vowel at idx={j}")
                return j
        # print(f"[Stress] Word '{word}': no vowel found, returning 0")
        return 0


@torch.inference_mode()
def generateText(
    model,
    tokens2id,
    auth,
    startSentence,
    limit=1000,
    temperature=0.4,
    K=16,
    max_line_len=80,
    w_rep=0.0,
    w_ll=10.0,
    w_rhyme=1.0,
    stress_predict=None,
    stress_dict=None,
    debug=False,
):
    """
    Char-level generator (single-pass per line; no repeated sampling).
    Assumes tokens2id maps single characters -> ids, and the model was trained with DP rhyme loss.
    """
    device = next(model.parameters()).device
    model.eval()

    V = len(tokens2id)
    id2tok = [""] * V
    for tok, i in tokens2id.items():
        id2tok[i] = tok

    space_id = tokens2id.get(" ", None)
    stop_id = tokens2id.get("}", None)
    nl_id = model.lineEndTokenIdx if getattr(model, "lineEndTokenIdx", None) is not None else tokens2id.get("\n")

    def is_cyrillic(ch: str) -> bool:
        return ('\u0400' <= ch <= '\u04FF') or ('\u0500' <= ch <= '\u052F')

    def tokens_to_text(toks):
        return "".join(id2tok[t] for t in toks)

    def extract_last_word(line_text: str) -> str:
        letters_only = "".join(c for c in line_text if is_cyrillic(c) or c == " ")
        parts = letters_only.split()
        return parts[-1] if parts else ""

    def extract_rime(line):
        # Operate on the full line to find the last Cyrillic word, as in generate_token.py
        line_text = line
        i = len(line_text) - 1
        while i >= 0:
            ch = line_text[i]
            if is_cyrillic(ch):
                break
            i -= 1
        if i < 0:
            return "", 0
        start = i
        while start >= 0:
            ch = line_text[start]
            if not is_cyrillic(ch):
                break
            start -= 1
        word = line_text[start + 1 : i + 1]
        if not word:
            return "", 0
        # Prefer passed-in stress_dict / stress_predict, then fallback
        sidx = None
        if stress_dict and word in stress_dict:
            if debug:
                print("[Gen] Found cached stress for word:", word)
            sidx = stress_dict[word]
        elif stress_predict is not None:
            if debug:
                print("[Gen] Predicting stress for word:", word)
            try:
                sidx = int(stress_predict(word))
            except Exception:
                sidx = None
        if sidx is None:
            sidx = get_stress_index(word)
        if sidx < 0 or sidx >= len(word):
            return "", 0
        tail = word[sidx:]
        return tail, sidx

    def step(last_id, h, c, last_tok):
        inp = torch.tensor([last_id], dtype=torch.long, device=device)
        E = model.embed(inp)
        o, (h2, c2) = model.lstm(E, hx=(h, c))
        logits = model.projection(o)[0]
        if last_tok == " " and space_id is not None:
            logits = logits.clone()
            logits[space_id] = -1e9
        logits = logits / max(1e-8, float(temperature))
        logp = F.log_softmax(logits, dim=-1)
        nxt = torch.multinomial(logp.exp(), 1).item()
        return nxt, float(logp[nxt].item()), h2, c2

    def force(last_id, h, c, last_tok, tid):
        inp = torch.tensor([last_id], dtype=torch.long, device=device)
        E = model.embed(inp)
        o, (h2, c2) = model.lstm(E, hx=(h, c))
        logits = model.projection(o)[0]
        if last_tok == " " and space_id is not None:
            logits = logits.clone()
            logits[space_id] = -1e9
        logp = F.log_softmax(logits, dim=-1)
        return int(tid), float(logp[int(tid)].item()), h2, c2

    # Char tokenization: just list the string
    seed_tokens = list(startSentence)
    prefix_ids = [tokens2id.get(t, model.unkTokenIdx) for t in seed_tokens]

    out_text = "".join(seed_tokens[1:]) if len(seed_tokens) > 0 else ""
    last_tok = seed_tokens[-1] if seed_tokens else ""
    last_id = prefix_ids[-1] if prefix_ids else (tokens2id.get("{", 0) if "{" in tokens2id else 0)

    # Init hidden state from author embeddings (unbatched LSTM mode)
    auth_id_t = torch.tensor(model.auth2id.get(auth, 0), dtype=torch.long, device=device)
    h = model.embed_auth_out(auth_id_t).unsqueeze(0).repeat(model.lstm_layers, 1)
    c = model.embed_auth_cell(auth_id_t).unsqueeze(0).repeat(model.lstm_layers, 1)

    # Feed the prefix
    if prefix_ids:
        inp = torch.tensor(prefix_ids, dtype=torch.long, device=device)
        E = model.embed(inp)
        _, (h, c) = model.lstm(E, hx=(h, c))
        last_id = int(prefix_ids[-1])

    base_tail = ""
    line_no = 0
    stress_positions = []

    def sample_one_line(h_start, c_start, last_id_start, last_tok_start):
        # Returns: toks, ll, h_end, c_end, ended_poem
        h_local = h_start.clone()
        c_local = c_start.clone()
        last_id_local = int(last_id_start)
        last_tok_local = last_tok_start

        toks = []
        ll = 0.0

        for _ in range(max_line_len):
            nxt, lp, h_local, c_local = step(last_id_local, h_local, c_local, last_tok_local)
            toks.append(nxt)
            ll += lp
            last_id_local = nxt
            last_tok_local = id2tok[nxt]
            if nxt == nl_id or nxt == stop_id:
                break

        # Force newline if needed
        if nl_id is not None and toks and toks[-1] != nl_id and toks[-1] != stop_id:
            nxt, lp, h_local, c_local = force(last_id_local, h_local, c_local, last_tok_local, nl_id)
            toks.append(nxt)
            ll += lp

        ended_poem = bool(stop_id is not None and toks and toks[-1] == stop_id)
        return toks, ll, h_local, c_local, ended_poem

    last_word_base = ""

    while len(out_text) < limit:
        if line_no % 2 == 0:
            # Generate base line (odd index human-readable)
            toks, ll, h_end, c_end, ended_poem = sample_one_line(h, c, last_id, last_tok)
            h, c = h_end, c_end
            if ended_poem:
                break

            line_text = tokens_to_text(toks[:-1])
            last_word = extract_last_word(line_text)
            tail, sidx = extract_rime(line_text)
            base_tail = tail
            last_word_base = last_word
            if debug:
                print(f"[Gen] Line {line_no+1} base tail: {repr(base_tail)} from last word {repr(last_word)}")

            stress_positions.append(sidx)
            out_text += tokens_to_text(toks)

            last_id = int(toks[-1])
            last_tok = id2tok[last_id]
        else:
            # Sample K candidates and pick best based on rhyme DP and repetition
            candidates = []
            for _ in range(max(1, int(K))):
                toks, ll, h_end, c_end, ended_poem = sample_one_line(h, c, last_id, last_tok)
                line_text = tokens_to_text(toks[:-1])
                cand_word = extract_last_word(line_text)
                cand_tail, sidx = extract_rime(line_text)

                # Normalize LL by length
                denom = max(1, len(toks))
                ll_norm = ll / denom
                rhyme_loss = rhyme_penalty_str(base_tail, cand_tail)
                rep_pen = repetition_penalty(toks[:-1], w_rep, id2tok)

                score = w_ll * ll_norm - w_rhyme * rhyme_loss - w_rep * rep_pen
                if last_word_base == cand_word and last_word_base != "":
                    score -= 20.0

                candidates.append((score, toks, h_end, c_end, ended_poem, sidx))

                # Keep candidates sorted for potential early pruning (optional)
                candidates.sort(key=lambda x: x[0], reverse=True)

            best_score, best_toks, best_h, best_c, ended_poem, best_sidx = candidates[0]
            if debug:
                print(f"[Gen] Line {line_no+1} best score={best_score:.4f}")

            stress_positions.append(best_sidx)
            out_text += tokens_to_text(best_toks)

            h, c = best_h, best_c
            if best_toks:
                last_id = int(best_toks[-1])
                last_tok = id2tok[last_id]

            if ended_poem:
                break

        line_no += 1

        if stop_id is not None and last_id == stop_id:
            break

    # Insert stress marks in the last word of each line (apostrophe), same as old generator
    final = ""
    lines = out_text.splitlines()
    for l, sidx in zip(lines, stress_positions):
        words = l.split(" ")
        if not words:
            final += "\n"
            continue
        count = 0
        last = ""
        for i in range(len(words[-1])):
            ch = words[-1][i]
            if is_cyrillic(ch):
                if count == sidx and sidx == 0:
                    last += "'"
                last += ch
                count += 1
                if count == sidx:
                    last += "'"
            else:
                last += ch
        result = words[:-1] + [last]
        final += " ".join(result) + "\n"

    return final
