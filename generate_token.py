import numpy as np
import torch
import torch.nn.functional as F
import math

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
                dp[i][j - 1] + 10.0,                # insertion
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


@torch.inference_mode()
def generateText(
    model,
    tokens2id,
    auth,
    startSentence,
    limit=1000,
    temperature=0.4,
    K=32,
    max_line_len=80,
    w_ll=10.0,
    w_rhyme=1.0,
    rhyme_k=3,
    w_rep=10,
    stress_predict=None,
    stress_dict=None,
    debug=False,
):
    device = next(model.parameters()).device
    model.eval()

    V = len(tokens2id)
    id2tok = [""] * V
    for tok, i in tokens2id.items():
        id2tok[i] = tok

    space_id = tokens2id.get(" ", None)
    stop_id = tokens2id.get("}", None)
    nl_id = model.lineEndTokenIdx or tokens2id.get("\n")

    def is_cyrillic(ch: str) -> bool:
        return ('\u0400' <= ch <= '\u04FF') or ('\u0500' <= ch <= '\u052F')

    def tokens_to_text(toks):
        return "".join(id2tok[t] for t in toks)

    def extract_last_word(line_text: str) -> str:
        letters_only = "".join(c for c in line_text if is_cyrillic(c) or c == " ")
        parts = letters_only.split()
        return parts[-1] if parts else ""

    def extract_rime(line):
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
        word = line_text[start+1:i+1]
        if not word:
            return "", 0
        if stress_dict and word in stress_dict:
            if debug: print("[Gen] Found cached stress for word:", word)
            sidx = stress_dict[word]
        elif stress_predict is not None:
            if debug: print("[Gen] Predicting stress for word:", word)
            sidx = int(stress_predict(word))
        else:
            sidx = 0
            for j in range(len(word) - 1, -1, -1):
                if word[j] in VOWELS_BG:
                    sidx = j
                    break
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
            logits = logits.clone(); logits[space_id] = -1e9
        logits /= max(1e-8, float(temperature))
        logp = F.log_softmax(logits, dim=-1)
        nxt = torch.multinomial(logp.exp(), 1).item()
        return nxt, float(logp[nxt].item()), h2, c2

    def force(last_id, h, c, last_tok, tid):
        inp = torch.tensor([last_id], dtype=torch.long, device=device)
        E = model.embed(inp)
        o, (h2, c2) = model.lstm(E, hx=(h, c))
        logits = model.projection(o)[0]
        if last_tok == " " and space_id is not None:
            logits = logits.clone(); logits[space_id] = -1e9
        logp = F.log_softmax(logits, dim=-1)
        return tid, float(logp[int(tid)].item()), h2, c2

    def tokenize_seed(s: str):
        toks = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '\n' or ch == ' ' or ch in ('{','}','|','@'):
                toks.append(ch); i += 1
            else:
                if ('\u0400' <= ch <= '\u04FF') or ('A' <= ch <= 'Z') or ('a' <= ch <= 'z'):
                    j = i + 1
                    while j < len(s):
                        cj = s[j]
                        if (('\u0400' <= cj <= '\u04FF') or ('A' <= cj <= 'Z') or ('a' <= cj <= 'z')):
                            j += 1
                        else:
                            break
                    toks.append(s[i:j]); i = j
                else:
                    toks.append(ch); i += 1
        return toks

    seed_tokens = tokenize_seed(startSentence)
    prefix_ids = [tokens2id.get(t, model.unkTokenIdx) for t in seed_tokens]
    out_text = "".join(seed_tokens[1:])
    last_tok = seed_tokens[-1]

    auth_id_t = torch.tensor(model.auth2id.get(auth, 0), dtype=torch.long, device=device)
    h = model.embed_auth_out(auth_id_t).unsqueeze(0).repeat(model.lstm_layers, 1)
    c = model.embed_auth_cell(auth_id_t).unsqueeze(0).repeat(model.lstm_layers, 1)

    with torch.no_grad():
        inp = torch.tensor(prefix_ids, dtype=torch.long, device=device)
        E = model.embed(inp)
        _, (h, c) = model.lstm(E, hx=(h, c))
        last_id = prefix_ids[-1]

    base_tail = ""
    line_no = 0
    stress_positions = []
    last_word_base = ""

    while len(out_text) < limit:
        if debug:
            print(f"[Gen] Generating line: {line_no+1}", end='\r', flush=True)
        def sample_line():
            toks, ll = [], 0.0
            h_local = h.clone(); c_local = c.clone()
            last_id_local = int(last_id); last_tok_local = last_tok
            prev_h = h_local; prev_c = c_local
            prev_last_id = last_id_local; prev_last_tok = last_tok_local
            forced_newline = False
            for _ in range(max_line_len):
                prev_h = h_local; prev_c = c_local
                prev_last_id = last_id_local; prev_last_tok = last_tok_local
                nxt, lp, h_local, c_local = step(last_id_local, h_local, c_local, last_tok_local)
                toks.append(nxt); ll += lp
                last_id_local = nxt; last_tok_local = id2tok[nxt]
                if nxt == nl_id or nxt == stop_id:
                    break
            if toks[-1] != nl_id:
                nxt, lp, h_local, c_local = force(last_id_local, h_local, c_local, last_tok_local, nl_id)
                toks.append(nxt); ll += lp
                forced_newline = True
            is_ended = False
            if stop_id is not None and len(toks) > 0 and (toks[-1] == stop_id or toks[0] == stop_id):
                is_ended = True

            if debug:
                dbg_last_id = last_id_local if forced_newline else prev_last_id
                dbg_last_tok = last_tok_local if forced_newline else prev_last_tok
                dbg_h = h_local if forced_newline else prev_h
                dbg_c = c_local if forced_newline else prev_c
                inp_dbg = torch.tensor([dbg_last_id], dtype=torch.long, device=device)
                E_dbg = model.embed(inp_dbg)
                o_dbg, _ = model.lstm(E_dbg, hx=(dbg_h, dbg_c))
                logits_dbg = model.projection(o_dbg)[0]
                if dbg_last_tok == " " and space_id is not None:
                    logits_dbg = logits_dbg.clone(); logits_dbg[space_id] = -1e9
                logits_dbg /= max(1e-8, float(temperature))
                probs_dbg = torch.softmax(logits_dbg, dim=-1)
                k = min(5, probs_dbg.shape[0])
                top_probs, top_idx = torch.topk(probs_dbg, k=k)
                top_items = [(id2tok[int(i)], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]
                print(f"[Debug] Final step top-{k}: " + ", ".join([f"{tok}:{prob:.4f}" for tok, prob in top_items]))
            return toks, ll, h_local, c_local, is_ended

        if line_no % 2 == 0:
            toks, ll, h2, c2, is_ended = sample_line()
            h, c = h2, c2
            if is_ended:
                break
            out_text += tokens_to_text(toks)
            
            line_text = tokens_to_text(toks[:-1])
            last_word_base = extract_last_word(line_text)
            base_tail, sidx = extract_rime(last_word_base)
            
            if debug:
                print(f"[First]    Line {line_no+1} generated: {''.join(id2tok[t] for t in toks)} | rhyme tail: {repr(base_tail)} | stress idx: {sidx}")
            stress_positions.append(sidx)
            last_id = toks[-1]; last_tok = id2tok[last_id]

            if debug:
                print(f"[First] Last word of line {line_no+1}: {repr(last_word_base)}")
        else:
            cands = []
            for _ in range(K):
                if debug:
                    print(f"==========================================================\n")
                toks, ll, _, _, is_ended = sample_line()
                
                cur_last_word = ""
                cur_line_text = tokens_to_text(toks[:-1])
                cur_last_word = extract_last_word(cur_line_text)
                cand_tail, sidx = extract_rime(cur_last_word)
                if debug:
                    print(f"[RHYME] Candidate last word: {repr(cur_last_word)} with tail {repr(cand_tail)}")
                
                ll /= max(1, len(toks))
                rhyme_loss = rhyme_penalty_str(base_tail, cand_tail)
                if debug:
                    print(f"[RHYME] Candidate rhyme loss: {rhyme_loss:.4f} for tail {repr(cand_tail)} against base tail {repr(base_tail)}")
                rep_pen = repetition_penalty(toks[:-1], w_rep, id2tok)
                
                score = w_ll * ll - w_rhyme * rhyme_loss - w_rep * rep_pen
                
                if debug:
                    print(f"[Gen] Candidate last word: {repr(cur_last_word)}")
                
                if last_word_base == cur_last_word and last_word_base != "":
                    score -= 20.0
                
                cands.append((score, toks, is_ended))
                
                if debug:
                    print(f"[Gen] Candidate for line {line_no+1}: {''.join(id2tok[t] for t in toks)} | ll={ll:.4f} | rhy={rhyme_loss:.4f} | rep={rep_pen:.4f} | score={score:.4f} | rhyme tail: {repr(cand_tail)}")

            best_score, best, is_ended = max(cands, key=lambda x: x[0])
            if debug:
                print(f"[Second] Selected for line {line_no+1}: {''.join(id2tok[t] for t in best)} | score={best_score:.4f}")
            if is_ended:
                break
            out_text += tokens_to_text(best)
            best_line_text = tokens_to_text(best[:-1])
            chosen_last_word = extract_last_word(best_line_text)
            _, sidx = extract_rime(chosen_last_word)
            stress_positions.append(sidx)
            last_id = best[-1]; last_tok = id2tok[last_id]
            
        line_no += 1
        if stop_id in prefix_ids:
            break

    print()
    final = ""
    lines = out_text.splitlines()
    for l, sidx in zip(lines, stress_positions):
        words = l.split(" ")
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
                    # print(f"[Gen] Stress position for line: {l} is at char index {i} (stress idx {sidx})")
                    last += "'"
            else:
                last += ch
        result = words[:-1] + [last]
        final += " ".join(result) + "\n"

    return final
