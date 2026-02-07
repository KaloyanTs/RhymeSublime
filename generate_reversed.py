import math
import csv
import torch
import torch.nn.functional as F
from stress import predict as stress_predict
from dp import rhyme_dp_penalty
try:
    from closeness import dist as phoneme_dist, letters as BG_LETTERS
except Exception:
    phoneme_dist = None
    BG_LETTERS = list("абвгдежзийклмнопрстуфхцчшщъьюя")


def reverse_each_line_keep_braces(text: str) -> str:
    """
    Reverse characters inside each line, keep line order.
    Keeps leading '{' and trailing '}' in place if present.
    """
    if not text:
        return text
    prefix = "{" if text.startswith("{") else ""
    suffix = "}" if text.endswith("}") else ""
    core = text[len(prefix): len(text) - len(suffix) if suffix else len(text)]
    parts = core.split("\n")
    core_rtl = "\n".join(p[::-1] for p in parts)
    return prefix + core_rtl + suffix


def rtl_to_ltr_poem(text_rtl: str) -> str:
    """Inverse of reverse_each_line_keep_braces, then strips braces if present."""
    if not text_rtl:
        return text_rtl
    s = text_rtl
    if s.startswith("{"):
        s = s[1:]
    if s.endswith("}"):
        s = s[:-1]
    parts = s.split("\n")
    return "\n".join(p[::-1] for p in parts)


VOWELS_BG = set(list("аеиоуъюяѝАЕИОУЪЮЯЍ"))
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
    sidx = i - 1  # backtick after the stressed char
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
        return _stress_index_from_stressed(stressed, word)

    try:
        return int(stress_predict(word))
    except Exception:
        for j in range(len(word) - 1, -1, -1):
            if word[j] in VOWELS_BG:
                return j
        return 0


def is_cyrillic(ch: str) -> bool:
    return ("\u0400" <= ch <= "\u04FF") or ("\u0500" <= ch <= "\u052F")


def extract_rime_ltr(line_text_ltr: str, *, stress_predict_fn=None, stress_dict=None):
    """
    Like your old extract_rime(): finds last Cyrillic word, returns (tail_from_stress, stress_idx).
    line_text_ltr MUST be normal left-to-right.
    """
    i = len(line_text_ltr) - 1
    while i >= 0 and not is_cyrillic(line_text_ltr[i]):
        i -= 1
    if i < 0:
        return "", 0

    start = i
    while start >= 0 and is_cyrillic(line_text_ltr[start]):
        start -= 1

    word = line_text_ltr[start + 1 : i + 1]
    if not word:
        return "", 0

    sidx = None
    if stress_dict and word in stress_dict:
        sidx = stress_dict[word]
    elif stress_predict_fn is not None:
        try:
            sidx = int(stress_predict_fn(word))
        except Exception:
            sidx = None
    if sidx is None:
        sidx = get_stress_index(word)

    if sidx < 0 or sidx >= len(word):
        return "", 0
    return word[sidx:], sidx


def add_stress_marks_last_word(poem_ltr: str, *, stress_predict_fn=None, stress_dict=None) -> str:
    """
    Inserts apostrophes in the last word of each line (same style as your old generator).
    """
    out_lines = []
    for line in poem_ltr.splitlines():
        tail, sidx = extract_rime_ltr(line, stress_predict_fn=stress_predict_fn, stress_dict=stress_dict)
        if not line:
            out_lines.append("")
            continue
        words = line.split(" ")
        if not words:
            out_lines.append(line)
            continue

        last_word = words[-1]
        count = 0
        marked = ""
        for ch in last_word:
            if is_cyrillic(ch):
                if count == sidx and sidx == 0:
                    marked += "'"
                marked += ch
                count += 1
                if count == sidx:
                    marked += "'"
            else:
                marked += ch

        out_lines.append(" ".join(words[:-1] + [marked]))
    return "\n".join(out_lines) + ("\n" if poem_ltr.endswith("\n") else "")


@torch.inference_mode()
def generateText_rtl_forced_rhyme(
    model,
    tokens2id: dict,
    auth: str,
    startSentence: str,
    *,
    limit: int = 1000,
    temperature: float = 0.6,
    max_line_len: int = 90,
    forbid_double_space: bool = True,
    forbid_line_start_space: bool = True,
    mark_stress: bool = True,
    stress_predict_fn=None,
    stress_dict=None,
    debug: bool = False,
    alpha_phonetic: float = 25.0,
    K: int = 1,
    w_rep: float = 0.0,
    w_ll: float = 10.0,
    w_rhyme: float = 1.0,
):
    """
    Model is trained on RTL-per-line text (each line reversed).
    Generation happens in RTL space, then we reverse each line back for normal output.

    Scheme:
      - Line 1 (odd): sample normally (RTL)
      - Extract rhyme tail from LTR version of Line 1
      - Line 2 (even): force reversed(tail) as the *beginning* of the RTL line (=> ending in LTR)
        then continue sampling the rest of the line normally
      - Repeat for (3,4), (5,6), ...

    No DP, no K-candidates.
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

    def tok2text(toks):
        return "".join(id2tok[t] for t in toks)

    def extract_last_word_ltr(line_text: str) -> str:
        letters_only = "".join(c for c in line_text if is_cyrillic(c) or c == " ")
        parts = letters_only.split()
        return parts[-1] if parts else ""

    def repetition_penalty(line_ids, w_rep_val: float, id2tok_map):
        if w_rep_val <= 0.0:
            return 0.0
        line_text = "".join(id2tok_map[t] for t in line_ids)
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

    # Prepare RTL prefix
    start_rtl = reverse_each_line_keep_braces(startSentence)
    seed_tokens = list(start_rtl)
    prefix_ids = [tokens2id.get(ch, model.unkTokenIdx) for ch in seed_tokens]

    # Output in RTL space (we skip the leading "{" like your old code did)
    out_rtl = "".join(seed_tokens[1:]) if seed_tokens else ""

    last_tok = seed_tokens[-1] if seed_tokens else ""
    last_id = prefix_ids[-1] if prefix_ids else (tokens2id.get("{", 0) if "{" in tokens2id else 0)

    # Init author state (unbatched mode, compatible with your current generator)
    auth_id_t = torch.tensor(model.auth2id.get(auth, 0), dtype=torch.long, device=device)
    h = model.embed_auth_out(auth_id_t).unsqueeze(0).repeat(model.lstm_layers, 1)
    c = model.embed_auth_cell(auth_id_t).unsqueeze(0).repeat(model.lstm_layers, 1)

    # Feed prefix through the LSTM
    if prefix_ids:
        inp = torch.tensor(prefix_ids, dtype=torch.long, device=device)
        E = model.embed(inp)
        _, (h, c) = model.lstm(E, hx=(h, c))
        last_id = int(prefix_ids[-1])
        last_tok = id2tok[last_id]

    # Reuse a 1-element tensor to reduce allocations
    one = torch.empty(1, dtype=torch.long, device=device)

    def _apply_space_rules(logits, prev_tok: str):
        if space_id is None:
            return logits
        if forbid_double_space and prev_tok == " ":
            logits = logits.clone()
            logits[space_id] = -1e9
            return logits
        if forbid_line_start_space and prev_tok == "\n":
            logits = logits.clone()
            logits[space_id] = -1e9
            return logits
        return logits

    def step_sample(prev_id: int, h0, c0, prev_tok: str):
        one[0] = prev_id
        E = model.embed(one)              # (1, E)
        o, (h1, c1) = model.lstm(E, hx=(h0, c0))
        logits = model.projection(o)[0]  # (V,)
        logits = _apply_space_rules(logits, prev_tok)
        logits = logits / max(1e-8, float(temperature))
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        logp = float(torch.log(probs[nxt] + 1e-12).item())
        if debug:
            try:
                ch = id2tok[nxt]
            except Exception:
                ch = "?"
            print(f"[Step] sample prev='{prev_tok}' id={prev_id} -> id={nxt} tok='{ch}' lp={logp:.4f}")
        return nxt, logp, h1, c1

    def step_force(prev_id: int, h0, c0, prev_tok: str, tid: int):
        one[0] = prev_id
        E = model.embed(one)
        o, (h1, c1) = model.lstm(E, hx=(h0, c0))
        logits = model.projection(o)[0]
        logits = _apply_space_rules(logits, prev_tok)
        logp = F.log_softmax(logits, dim=-1)
        if debug:
            tok = id2tok[tid] if 0 <= tid < len(id2tok) else str(tid)
            lp = float(logp[tid].item()) if 0 <= tid < logp.numel() else float('nan')
            print(f"[Step] force prev='{prev_tok}' id={prev_id} -> id={tid} tok='{tok}' lp={lp:.4f}")
        return tid, float(logp[tid].item()), h1, c1

    letter_id_map = {}
    for ch in BG_LETTERS:
        if ch in tokens2id:
            letter_id_map[ch] = tokens2id[ch]

    def step_force_phonetic(prev_id: int, h0, c0, prev_tok: str, target_char: str):
        one[0] = prev_id
        E = model.embed(one)
        o, (h1, c1) = model.lstm(E, hx=(h0, c0))
        logits = model.projection(o)[0]
        logits = _apply_space_rules(logits, prev_tok)
        # Build distribution over letter tokens; if target is non-letter, keep exact force if available
        t_lower = target_char.lower()
        if phoneme_dist is None or t_lower not in BG_LETTERS:
            tid = tokens2id.get(target_char, model.unkTokenIdx)
            logp = F.log_softmax(logits, dim=-1)
            if debug:
                tok = id2tok[tid] if 0 <= tid < len(id2tok) else str(tid)
                lp = float(logp[tid].item()) if 0 <= tid < logp.numel() else float('nan')
                print(f"[Step] force(no-phoneme) prev='{prev_tok}' id={prev_id} target='{target_char}' -> id={tid} tok='{tok}' lp={lp:.4f}")
            return tid, float(logp[tid].item()), h1, c1

        candidates = []
        weights = []
        for cand_char, cand_id in letter_id_map.items():
            c_lower = cand_char
            d = float(phoneme_dist(t_lower, c_lower)) if phoneme_dist else (0.0 if t_lower == c_lower else 1.0)
            w = math.exp(-float(alpha_phonetic) * d)
            candidates.append(int(cand_id))
            weights.append(w)
        if not candidates:
            tid = tokens2id.get(target_char, model.unkTokenIdx)
        else:
            S = sum(weights)
            if S <= 0:
                tid = tokens2id.get(target_char, model.unkTokenIdx)
            else:
                probs = torch.tensor([w / S for w in weights], dtype=torch.float, device=logits.device)
                idx = torch.multinomial(probs, 1).item()
                tid = int(candidates[idx])
        logp = F.log_softmax(logits, dim=-1)
        if debug:
            tok = id2tok[tid] if 0 <= tid < len(id2tok) else str(tid)
            lp = float(logp[tid].item()) if 0 <= tid < logp.numel() else float('nan')
            print(f"[Step] force(phoneme) prev='{prev_tok}' id={prev_id} target='{target_char}' -> id={tid} tok='{tok}' lp={lp:.4f}")
        return tid, float(logp[tid].item()), h1, c1

    def sample_line(h0, c0, prev_id: int, prev_tok: str, forced_prefix_ids=None, forced_prefix_chars=None):
        """
        Returns (toks_including_newline_or_stop, ll_sum, h_end, c_end, ended_poem)
        """
        h = h0.clone()
        c = c0.clone()
        pid = int(prev_id)
        ptok = prev_tok

        toks = []
        ll = 0.0

        if debug:
            fp_ids_len = len(forced_prefix_ids) if forced_prefix_ids else 0
            fp_chars = ''.join(forced_prefix_chars) if forced_prefix_chars else ''
            print(f"[Line] start prev_tok='{ptok}' prev_id={pid} forced_ids={fp_ids_len} forced_chars='{fp_chars}'")

        # Force prefix first (this becomes the LTR ending after final per-line reverse)
        if forced_prefix_ids or forced_prefix_chars:
            if forced_prefix_chars:
                for ch in forced_prefix_chars:
                    tid, _, h, c = step_force_phonetic(pid, h, c, ptok, ch)
                    toks.append(tid)
                    pid = tid
                    ptok = id2tok[tid]
                    if tid == nl_id or tid == stop_id:
                        if debug:
                            print(f"[Line] early-end on forced char '{ch}' tid={tid}")
                        return toks, ll, h, c, (tid == stop_id)
            else:
                for tid in forced_prefix_ids:
                    tid, _, h, c = step_force(pid, h, c, ptok, int(tid))
                    toks.append(tid)
                    pid = tid
                    ptok = id2tok[tid]
                    if tid == nl_id or tid == stop_id:
                        if debug:
                            print(f"[Line] early-end on forced id tid={tid}")
                        return toks, ll, h, c, (tid == stop_id)

        budget = max(1, int(max_line_len) - len(toks))
        if debug:
            print(f"[Line] sampling budget={budget}")
        for _ in range(budget):
            nxt, lp, h, c = step_sample(pid, h, c, ptok)
            toks.append(nxt)
            ll += lp
            pid = nxt
            ptok = id2tok[nxt]
            if nxt == nl_id or nxt == stop_id:
                if debug:
                    print(f"[Line] break on token id={nxt} tok='{ptok}'")
                break

        if nl_id is not None and toks and toks[-1] not in (nl_id, stop_id):
            tid, lp, h, c = step_force(pid, h, c, ptok, int(nl_id))
            toks.append(tid)
            ll += lp
            pid = tid
            ptok = id2tok[tid]
            if debug:
                print(f"[Line] forced newline tid={tid}")

        ended_poem = bool(stop_id is not None and toks and toks[-1] == stop_id)
        if debug:
            line_rtl_dbg = tok2text(toks)
            line_ltr_dbg = line_rtl_dbg[::-1]
            print(f"[Line] end len={len(toks)} ll={ll:.4f} ended={ended_poem} rtl='{line_rtl_dbg}' ltr='{line_ltr_dbg}'")
        return toks, ll, h, c, ended_poem

    line_no = 0
    base_tail_ltr = ""
    base_last_word = ""

    while len(out_rtl) < int(limit):
        if debug:
            print(f"[Loop] line_no={line_no+1} mode={'odd' if line_no % 2 == 0 else 'even'}")
        forced = None
        forced_chars = None
        if line_no % 2 == 0:
            forced = None
            toks, ll, h, c, ended = sample_line(
                h, c, last_id, last_tok,
                forced_prefix_ids=forced,
                forced_prefix_chars=forced_chars,
            )
        else:
            forced = None
            forced_chars = None
            if base_tail_ltr:
                forced_tail_rtl = base_tail_ltr[::-1]
                forced_chars = list(forced_tail_rtl)
                if debug:
                    print(f"[Loop] forced_tail_ltr='{base_tail_ltr}' forced_prefix_rtl='{forced_tail_rtl}'")

            if int(K) <= 1:
                toks, ll, h, c, ended = sample_line(
                    h, c, last_id, last_tok,
                    forced_prefix_ids=forced,
                    forced_prefix_chars=forced_chars,
                )
            else:
                best = None
                best_pen = None
                best_pack = None
                tk = None
                for k in range(int(K)):
                    tk, ll_k, hk, ck, endk = sample_line(
                        h, c, last_id, last_tok,
                        forced_prefix_ids=forced,
                        forced_prefix_chars=forced_chars,
                    )
                    line_rtl_k = tok2text(tk[:-1]) if tk and tk[-1] in (nl_id, stop_id) else tok2text(tk)
                    line_ltr_k = line_rtl_k[::-1]
                    cand_tail_k, _ = extract_rime_ltr(
                        line_ltr_k, stress_predict_fn=stress_predict_fn, stress_dict=stress_dict
                    )
                    cand_word_k = extract_last_word_ltr(line_ltr_k)
                    denom = max(1, len(tk))
                    ll_norm = ll_k / denom
                    rhyme_loss = float(rhyme_dp_penalty(base_tail_ltr or "", cand_tail_k or ""))
                    rep_pen = repetition_penalty(tk[:-1], w_rep, id2tok)
                    score = float(w_ll) * ll_norm - float(w_rhyme) * rhyme_loss - float(w_rep) * rep_pen
                    if base_last_word and cand_word_k and base_last_word == cand_word_k:
                        score -= 20.0
                    if debug:
                        print(f"[GenRTL] K={k+1}/{K} word={repr(cand_word_k)} cand_tail={repr(cand_tail_k)} ll_norm={ll_norm:.4f} rhyme={rhyme_loss:.4f} rep={rep_pen:.4f} score={score:.4f}")
                    if best is None or score > best[0]:
                        best = (score, tk)
                        best_pack = (hk, ck, endk)
                if debug and best is not None:
                    print(f"[GenRTL] chosen score={best[0]:.4f}")
                toks = best[1] if best is not None else tk
                h, c, ended = best_pack if best_pack is not None else (h, c, False)
        if not toks:
            break

        last_id = int(toks[-1])
        last_tok = id2tok[last_id]

        out_rtl += tok2text(toks)

        line_rtl = tok2text(toks[:-1]) if toks[-1] in (nl_id, stop_id) else tok2text(toks)
        line_ltr = line_rtl[::-1]
        if line_no % 2 == 0:
            base_tail_ltr, _ = extract_rime_ltr(
                line_ltr, stress_predict_fn=stress_predict_fn, stress_dict=stress_dict
            )
            base_last_word = extract_last_word_ltr(line_ltr)
            if debug:
                print(f"[GenRTL] Line {line_no+1} base tail (LTR): {repr(base_tail_ltr)} last_word={repr(base_last_word)}")

        line_no += 1
        if ended or (stop_id is not None and last_id == stop_id):
            break

    poem_ltr = rtl_to_ltr_poem(out_rtl)
    if mark_stress:
        poem_ltr = add_stress_marks_last_word(
            poem_ltr, stress_predict_fn=stress_predict_fn, stress_dict=stress_dict
        )
    return poem_ltr