import numpy as np
import torch
import torch.nn.functional as F
import time

VOWELS_BG = set(list("аеиоуъюяѝАЕИОУЪЮЯЍ"))

OPPOSITE = {
    "б": "п", "п": "б",
    "в": "ф", "ф": "в",
    "г": "к", "к": "г",
    "д": "т", "т": "д",
    "ж": "ш", "ш": "ж",
    "з": "с", "с": "з",
    "Б": "П", "П": "Б",
    "В": "Ф", "Ф": "В",
    "Г": "К", "К": "Г",
    "Д": "Т", "Т": "Д",
    "Ж": "Ш", "Ш": "Ж",
    "З": "С", "С": "З",
}

def rhyme_bonus(base_rime_ids, cand_rime_ids, id2char):

    if not base_rime_ids or not cand_rime_ids:
        return 0.0

    a = base_rime_ids
    b = cand_rime_ids
    n, m = len(a), len(b)

    def is_vowel(tid):
        ch = id2char[tid] if 0 <= tid < len(id2char) else ""
        return ch in VOWELS_BG
    def del_cost(tid):
        return 3 if is_vowel(tid) else 1
    def ins_cost(tid):
        return 3 if is_vowel(tid) else 1
    def sub_cost(tid1, tid2):
        if tid1 == tid2:
            return 0.0
        if OPPOSITE.get(id2char[tid1], None) == id2char[tid2]:
            return 0.2
        return 3 if (is_vowel(tid1) or is_vowel(tid2)) else 1

    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + ins_cost(b[j - 1])

    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)
        cur[0] = prev[0] + del_cost(a[i - 1])
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j] + del_cost(a[i - 1]),
                cur[j - 1] + ins_cost(b[j - 1]),
                prev[j - 1] + sub_cost(a[i - 1], b[j - 1])
            )
        prev = cur

    dist = prev[m]
    if base_rime_ids[0] != cand_rime_ids[0]:
        dist += 10.0  # penalty for not matching stressed char
    if is_vowel(base_rime_ids[-1]) and not is_vowel(cand_rime_ids[-1]):
        dist += 5.0  # penalty for not matching vowel ending
    if not is_vowel(base_rime_ids[-1]) and is_vowel(cand_rime_ids[-1]):
        dist += 5.0  # penalty for not matching vowel ending
    return -float(dist)


def repetition_penaly(line_ids_no_nl, w_rep, id2char):
    if w_rep <= 0.0:
        return 0.0
    pen = 0.0
    run = 1
    for i in range(1, len(line_ids_no_nl)):
        if line_ids_no_nl[i] == line_ids_no_nl[i - 1]:
            run += 1
            if run >= 2:
                pen += 0.5 if VOWELS_BG.__contains__(id2char[line_ids_no_nl[i]]) else 1.0
        else:
            run = 1
    return pen

with torch.inference_mode():
    def generateText(
        model,
        char2id,
        auth,
        startSentence,
        limit=1000,
        temperature=0.4,
        K=64,
        max_line_len=80,
        w_ll=3.0,
        w_rhyme=0.5,
        rhyme_k=3,
        w_rep=10,
        stress_predict=None,
        stress_dict=None,
        stats=False
    ):
        device = next(model.parameters()).device
        model.eval()

        # id -> char
        V = len(char2id)
        id2char = [""] * V
        for ch, i in char2id.items():
            if 0 <= i < V:
                id2char[i] = ch

        space_id = char2id.get(" ", None)
        stop_id = char2id.get("}", None)
        nl_id = getattr(model, "lineEndTokenIdx", None) or char2id.get("\n", None)
        if nl_id is None:
            raise ValueError("No newline token found (model.lineEndTokenIdx and char2id['\\n'] missing).")

        auth_id = torch.tensor(model.auth2id.get(auth, 0), dtype=torch.long, device=device)
        h = model.embed_auth_out(auth_id).unsqueeze(0).repeat(model.lstm_layers, 1)
        c = model.embed_auth_cell(auth_id).unsqueeze(0).repeat(model.lstm_layers, 1)

        if not startSentence or startSentence[0] != "{":
            raise ValueError("startSentence must start with '{'.")

        out = startSentence[1:]
        
        line_times = []
        stress_dict_times = []
        stress_predict_times = []

        def is_cyrillic_letter(ch: str) -> bool:
            return ('\u0400' <= ch <= '\u04FF') or ('\u0500' <= ch <= '\u052F')

        def extract_rime_ids_from_line_ids(line_ids_no_nl, id2char, char2id, unk_id, rhyme_k, stress_predict):
            i = len(line_ids_no_nl) - 1
            while i >= 0:
                ch = id2char[line_ids_no_nl[i]] if 0 <= line_ids_no_nl[i] < len(id2char) else ""
                if is_cyrillic_letter(ch):
                    break
                i -= 1
            if i < 0:
                return [], 0

            word_rev = []
            while i >= 0:
                ch = id2char[line_ids_no_nl[i]] if 0 <= line_ids_no_nl[i] < len(id2char) else ""
                if not is_cyrillic_letter(ch):
                    break
                word_rev.append(ch)
                i -= 1

            word = "".join(reversed(word_rev))
            if not word:
                return [], 0
            
            sidx = rhyme_k

            if stress_predict is None:
                tail = word[-rhyme_k:] if rhyme_k and rhyme_k > 0 else word
            else:
                # print("Extracting rhyme IDs for word:", word)
                
                sidx = -1
                if stress_dict is not None and word in stress_dict:
                    t0 = time.perf_counter()
                    sidx = stress_dict[word]
                    stress_dict_times.append(time.perf_counter() - t0)
                    # print(f"  Found in dictionary: {word} -> stress index {sidx}")
                else:
                    t0 = time.perf_counter()
                    sidx = int(stress_predict(word))
                    stress_predict_times.append(time.perf_counter() - t0)
                    # print(f"  Predicted: {word} -> stress index {sidx}")
                
                if sidx < 0 or sidx >= len(word):
                    return [], 0
                tail = word[sidx:]
                if rhyme_k and rhyme_k > 0 and len(tail) > rhyme_k:
                    tail = tail[-rhyme_k:]

            return ([char2id.get(ch, unk_id) for ch in tail],sidx)

        @torch.no_grad()
        def step(last_id_local, h_local, c_local, last_char_local, line_no):
            inp = torch.tensor([last_id_local], dtype=torch.long, device=device)  # (1,)
            currentModel = model # TODO: Idea for second no=rhyming model
            E = currentModel.embed(inp)  # (1,E)
            o, (h2, c2) = currentModel.lstm(E, hx=(h_local, c_local))
            logits = currentModel.projection(o)[0]  # (V,)

            if last_char_local == " " and space_id is not None:
                logits = logits.clone()
                logits[space_id] = -1e9

            logits = logits / max(1e-8, float(temperature))
            logp = F.log_softmax(logits, dim=-1)
            nxt = torch.multinomial(logp.exp(), 1).item()
            return int(nxt), float(logp[nxt].item()), h2, c2

        @torch.no_grad()
        def force_token(last_id_local, h_local, c_local, last_char_local, forced_id):
            inp = torch.tensor([last_id_local], dtype=torch.long, device=device)
            E = model.embed(inp)
            o, (h2, c2) = model.lstm(E, hx=(h_local, c_local))
            logits = model.projection(o)[0]

            if last_char_local == " " and space_id is not None:
                logits = logits.clone()
                logits[space_id] = -1e9

            logits = logits / max(1e-8, float(temperature))
            logp = F.log_softmax(logits, dim=-1)
            return int(forced_id), float(logp[int(forced_id)].item()), h2, c2

        @torch.no_grad()
        def sample_one_line(h_start, c_start, last_id_start, last_char_start, line_no):
            h_local = h_start.clone()
            c_local = c_start.clone()
            last_id_local = int(last_id_start)
            last_char_local = last_char_start

            toks = []
            ll = 0.0

            for _ in range(max_line_len):
                nxt, logp, h_local, c_local = step(last_id_local, h_local, c_local, last_char_local, line_no)

                if stop_id is not None and nxt == stop_id:
                    return toks, ll, h_local, c_local, True  # ended poem (stop not included)

                toks.append(nxt)
                ll += logp
                last_id_local = nxt
                last_char_local = id2char[nxt] if 0 <= nxt < len(id2char) else ""

                if nxt == nl_id:
                    return toks, ll, h_local, c_local, False

            # New line must be forced if not yet ended
            nxt, logp, h_local, c_local = force_token(last_id_local, h_local, c_local, last_char_local, nl_id)
            toks.append(nxt)
            ll += logp
            return toks, ll, h_local, c_local, False

        prefix_ids = [char2id.get(ch, model.unkTokenIdx) for ch in startSentence]
        with torch.no_grad():
            inp = torch.tensor(prefix_ids, dtype=torch.long, device=device)
            E = model.embed(inp)
            _, (h, c) = model.lstm(E, hx=(h, c))
            last_id = prefix_ids[-1]
            last_char = out[-1] if out else None

        line_no = 0
        base_rime_ids = []
        stress_positions = []
        ended_poem = False

        while len(out) < limit and not ended_poem:
            print("Generating line number", line_no + 1, end="\r", flush=True)
            if line_no % 2 == 0:
                line_start_time = time.perf_counter()
                toks, ll, h_end, c_end, ended_poem = sample_one_line(h, c, last_id, last_char, line_no)
                line_times.append(time.perf_counter() - line_start_time)

                for tid in toks:
                    out += id2char[tid]
                    if len(out) >= limit or ended_poem:
                        break
                h, c = h_end, c_end
                if toks:
                    last_id = toks[-1]
                    last_char = id2char[last_id]

                if ended_poem:
                    break

                line_no_nl = toks[:-1] if toks and toks[-1] == nl_id else toks
                while line_no_nl and not is_cyrillic_letter(id2char[line_no_nl[-1]] if 0 <= line_no_nl[-1] < len(id2char) else ""):
                    line_no_nl = line_no_nl[:-1]
                base_rime_ids, stress_position = extract_rime_ids_from_line_ids(
                    line_no_nl, id2char, char2id, model.unkTokenIdx, rhyme_k, stress_predict
                )
                
                stress_positions.append(stress_position)
                
                # print("[odd]  Base rhyme ending for:", "".join(id2char[tid] for tid in line_no_nl), "is", "".join(id2char[tid] for tid in base_rime_ids),"stress_position:", stress_position)
            else:
                # print("Generating rhyming line for line", "".join(id2char[tid] for tid in base_rime_ids))
                
                line_start_time = time.perf_counter()
                candidates = []
                
                for _ in range(K):
                    toks, ll, h_end, c_end, ended_poem = sample_one_line(h, c, last_id, last_char, line_no)
                    line_no_nl = toks[:-1] if toks and toks[-1] == nl_id else toks

                    cand_rime_ids, cand_stress_position = extract_rime_ids_from_line_ids(
                        line_no_nl, id2char, char2id, model.unkTokenIdx, rhyme_k, stress_predict
                    )
                    
                    denom = max(1, len(toks))
                    rhyme_bonus_val = rhyme_bonus(base_rime_ids, cand_rime_ids, id2char)
                    repetition_penaly_val = repetition_penaly(line_no_nl, w_rep, id2char)
                    score = (
                        w_ll * (ll / denom)
                        + w_rhyme * rhyme_bonus_val
                        - w_rep * repetition_penaly_val
                    )
                    
                    # print("Candidate ending:", "".join(id2char[tid] for tid in cand_rime_ids), "Score:", w_ll * (ll / denom), "+", w_rhyme * rhyme_bonus_val, "-", w_rep * repetition_penaly_val, "=", score)
                    
                    candidates.append((score, toks, h_end, c_end, ended_poem, cand_rime_ids, cand_stress_position))

                    candidates.sort(key=lambda x: x[0], reverse=True)
                _, best_toks, best_h, best_c, ended_poem, best_rime_ids, best_stress_position = candidates[0]
                line_times.append(time.perf_counter() - line_start_time)
                
                stress_positions.append(best_stress_position)
                
                # print("[even] Base rhyme ending for:", "".join(id2char[tid] for tid in best_toks), "is", "".join(id2char[tid] for tid in best_rime_ids),"stress_position:", best_stress_position)

                for tid in best_toks:
                    out += id2char[tid]
                    if len(out) >= limit or ended_poem:
                        break

                h, c = best_h, best_c
                if best_toks:
                    last_id = best_toks[-1]
                    last_char = id2char[last_id]

                if ended_poem:
                    break

            line_no += 1
        
        # Print timing statistics
        if line_times and stats:
            avg_line_time = sum(line_times) / len(line_times)
            print(f"\n=== Timing Statistics ===")
            print(f"Average time per line: {avg_line_time*1000:.2f} ms ({len(line_times)} lines)")
        
        if stress_dict_times and stats:
            avg_dict_time = sum(stress_dict_times) / len(stress_dict_times)
            print(f"Average dictionary lookup time: {avg_dict_time*1000:.4f} ms ({len(stress_dict_times)} lookups)")
        
        if stress_predict_times and stats:
            avg_predict_time = sum(stress_predict_times) / len(stress_predict_times)
            print(f"Average stress prediction time: {avg_predict_time*1000:.2f} ms ({len(stress_predict_times)} predictions)")
        
        if stats:
            total_stress_ops = len(stress_dict_times) + len(stress_predict_times)
            if total_stress_ops > 0:
                dict_pct = (len(stress_dict_times) / total_stress_ops) * 100
                predict_pct = (len(stress_predict_times) / total_stress_ops) * 100
                print(f"Stress determination: {len(stress_dict_times)} dict ({dict_pct:.1f}%), {len(stress_predict_times)} predicted ({predict_pct:.1f}%)")
            print("=" * 25)
            
        final = ""
        print()
        # print("Stress positions:", stress_positions)
        for l,stress_position in zip(out.splitlines(), stress_positions):
            # print("Line:", l, "Stress position in rhyme ending:", stress_position)
            words = l.split(" ")
            count = 0
            last = ""
            for i in range(len(words[-1])):
                if is_cyrillic_letter(words[-1][i]):
                    if count == stress_position and stress_position == 0:
                        last += "'"
                    last += words[-1][i]
                    count += 1
                    if count == stress_position:
                        last += "'"
                
            result = words[:-1] + [last]
            final += " ".join(result) + "\n"

        return final