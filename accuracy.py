import sys
import random
import pickle
import argparse

import generate_token
import generate_char_dp
import model_token
import model_char_dp
import stress

from dp import rhyme_dp_penalty

from parameters import (
    device,
    char_emb_size, hid_size, lstm_layers, dropout,
    modelFileName, char2idFileName,
    # token
    char_emb_size_token, hid_size_token, lstm_layers_token, dropout_token,
    modelFileName_token, tokens2idFileName_token,
    # char-dp
    char_emb_size_dp, hid_size_dp, lstm_layers_dp, dropout_dp,
    modelFileName_dp, tokens2idFileName_dp,
    auth2idFileName,
    defaultTemperature,
)

from generate_char_dp import rhyme_penalty_str
import parameters

VOWELS_BG = set(list("аеиоуъюяѝАЕИОУЪЮЯЍ"))

def is_cyrillic(ch: str) -> bool:
    return ('\u0400' <= ch <= '\u04FF') or ('\u0500' <= ch <= '\u052F')

def extract_last_word(line_text: str) -> str:
    letters_only = "".join(c for c in line_text if is_cyrillic(c) or c == " ")
    parts = letters_only.split()
    return parts[-1] if parts else ""

def load_stress_dict(csv_path: str = 'bg_dict_csv/single_stress.csv'):
    """Load stress dictionary mapping plain word -> stress index (int)."""
    try:
        import csv as _csv
        stress_dict = {}
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = _csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # Expect header: id,name,name_stressed
                if row[0].strip().lower() == 'id':
                    continue
                if len(row) >= 3:
                    name = str(row[1]).strip()
                    stressed = str(row[2]).strip()
                    if name:
                        i = stressed.find('`')
                        if i >= 1:
                            stress_dict[name] = i - 1
        return stress_dict
    except Exception:
        return {}

def extract_tail_stress(word: str, stress_dict: dict):
    if not word:
        return "", 0
    # Prefer dictionary, then model; fallback to last vowel
    if word in stress_dict:
        sidx = int(stress_dict[word])
    else:
        try:
            sidx = int(stress.predict(word))
        except Exception:
            sidx = 0
            for j in range(len(word) - 1, -1, -1):
                if word[j] in VOWELS_BG:
                    sidx = j
                    break
    if sidx < 0 or sidx >= len(word):
        return "", 0
    tail = word[sidx:]
    return tail, sidx


def load_env(model_type: str, debug: bool = False):
    mt = model_type.lower()
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    stress_dict = load_stress_dict('bg_dict_csv/single_stress.csv')
    if debug:
        print(f"[Accuracy] Loaded stress_dict entries: {len(stress_dict)}")
        
    if mt == 'token':
        tokens2id = pickle.load(open(tokens2idFileName_token, 'rb'))
        lm = model_token.TokenLSTMLanguageModelPack(
            char_emb_size_token, hid_size_token, auth2id, tokens2id,
            unkToken='@', padToken='|', endToken='}',
            lstm_layers=lstm_layers_token, dropout=dropout_token,
            lambda_rhyme=0.1,
        ).to(device)
        lm.load(modelFileName_token, device)
        gen_fn = lambda auth, seed, K, temp: generate_token.generateText(
            lm, tokens2id, auth, seed, temperature=temp, K=K, stress_predict=stress.predict, stress_dict=stress_dict, debug=debug
        )
        return lm, tokens2id, auth2id, gen_fn, stress_dict
    elif mt == 'char_dp':
        tokens2id = pickle.load(open(tokens2idFileName_dp, 'rb'))
        lm = model_char_dp.CharLSTMLanguageModelPack(
            char_emb_size_dp, hid_size_dp, auth2id, tokens2id,
            unkToken='@', padToken='|', endToken='}',
            lstm_layers=lstm_layers_dp, dropout=dropout_dp,
            lambda_rhyme=0.0,
        ).to(device)
        lm.load(modelFileName_dp, device)
        gen_fn = lambda auth, seed, K, temp: generate_char_dp.generateText(
            lm, tokens2id, auth, seed, temperature=temp, K=K, stress_predict=stress.predict, stress_dict=stress_dict, debug=debug
        )
        return lm, tokens2id, auth2id, gen_fn, stress_dict
    else:
        raise ValueError("Unsupported model_type. Use one of: token, char_dp")


def compute_pair_penalties(poem_text: str, stress_dict: dict, debug: bool = False):
    """Return (penalties list, number of non-empty lines) for poem pairs (odd, even)."""
    lines = [ln for ln in poem_text.splitlines() if ln.strip() != ""]
    penalties = []
    for i in range(1, len(lines), 2):
        base = lines[i - 1]
        cand = lines[i]
        base_last = extract_last_word(base)
        cand_last = extract_last_word(cand)
        base_tail, _ = extract_tail_stress(base_last, stress_dict)
        cand_tail, _ = extract_tail_stress(cand_last, stress_dict)
        if base_tail and cand_tail:
            pen = float(rhyme_dp_penalty(base_tail, cand_tail))
            if debug:
                print(f"[Pair] base_last={repr(base_last)} tail={repr(base_tail)} | cand_last={repr(cand_last)} tail={repr(cand_tail)} | penalty={pen:.4f}")
            penalties.append(pen)
    return penalties, len(lines)


def run_for_model(mt: str, K: int, N_lines: int, seed: int, debug: bool = False):
    """Generate until total sampled lines >= N_lines; average over all pair penalties."""
    _, _, auth2id, gen_fn, stress_dict = load_env(mt, debug=debug)
    authors = list(auth2id.keys())
    if not authors:
        print("No authors found in auth2id.")
        return float("nan"), 0, 0
    random.seed(seed)
    total_lines = 0
    total_pairs = 0
    total_penalty = 0.0
    start_seed = '{'
    while total_lines < N_lines:
        auth = random.choice(authors)
        try:
            if debug:
                print(f"[Gen] model={mt} K={K} author={auth}")
            print("Total lines so far:", total_lines, end='\r', flush=True)
            poem = gen_fn(auth, start_seed, K, defaultTemperature)
            if debug:
                print("[Accuracy] Poem output:\n" + poem)
            pen_list, lines_used = compute_pair_penalties(poem, stress_dict, debug=debug)
            if debug:
                print(f"[Accuracy] Generated lines this pass: {lines_used}")
            total_lines += lines_used
            total_pairs += len(pen_list)
            total_penalty += sum(pen_list)
        except Exception:
            continue
    if total_pairs <= 0:
        return float("nan"), 0, total_lines
    return (total_penalty / total_pairs), total_pairs, total_lines


def main():
    parser = argparse.ArgumentParser(description="Compute average rhyme DP penalty across random generated poems.")
    parser.add_argument("model", nargs="?", help="Single model type: char|token|char_dp (deprecated if --models provided)")
    parser.add_argument("K", nargs="?", type=int, help="Sampling K (deprecated if --models provided)")
    parser.add_argument("N", nargs="?", type=int, default=1000, help="Line budget N: stop when total lines >= N (default 1000)")
    parser.add_argument("--models", nargs="+", help="List of model types to evaluate (char token char_dp)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for author sampling")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument("--out", type=str, help="Output CSV filename (defaults to <models>.res or <model>.res)")
    args = parser.parse_args()

    K_VALUES = [1, 2, 4, 8, 16, 32, 64]

    if args.models:
        N = args.N if isinstance(args.N, int) else 1000
        results = []
        for mt in args.models:
            for K in K_VALUES:
                avg_pen, n_pairs, n_lines = run_for_model(mt, K, N, args.seed, debug=args.debug)
                print(f"model={mt} K={K} avg_rhyme_penalty={avg_pen:.4f} (pairs={n_pairs}, lines={n_lines})")
                results.append({
                    'model': mt,
                    "rhyme_loss": parameters.lambda_rhyme_token if mt == 'token' else (parameters.lambda_rhyme_dp if mt == 'char_dp' else parameters.lambda_rhyme),
                    'K': K,
                    'avg_rhyme_penalty': avg_pen,
                    'pairs': n_pairs,
                    'lines': n_lines,
                })
        out_name = args.out if args.out else f"{'_'.join(args.models)}.res"
        try:
            with open(out_name, 'w', encoding='utf-8') as f:
                f.write("model,K,avg_rhyme_penalty,pairs,lines\n")
                for r in results:
                    f.write(f"{r['model']},{r['K']},{r['avg_rhyme_penalty']:.6f},{r['pairs']},{r['lines']}\n")
            print(f"[Save] Wrote consolidated results to {out_name}")
        except Exception as e:
            print(f"Warning: failed to save results to {out_name}: {e}")
        print("[Final] Average rhyme penalty per model and K:")
        by_model = {}
        for r in results:
            by_model.setdefault(r['model'], []).append((r['K'], r['avg_rhyme_penalty']))
        for mt, items in by_model.items():
            items.sort(key=lambda x: x[0])
            print(f"  model={mt}")
            for K, avg in items:
                print(f"    K={K}: {avg:.6f}")
        return

    # Fallback: single-model positional mode
    if not args.model or args.K is None:
        print("Usage: python accuracy.py --models <char token char_dp> [--N 1000] [--seed 42]\n       or: python accuracy.py <model_type> <K> [N]")
        sys.exit(1)

    mt = args.model
    K = int(args.K)
    N = int(args.N) if isinstance(args.N, int) else 1000
    avg_pen, n_pairs, n_lines = run_for_model(mt, K, N, args.seed, debug=args.debug)
    print(f"model={mt} K={K} avg_rhyme_penalty={avg_pen:.4f} (pairs={n_pairs}, lines={n_lines})")
    # Save single-model result
    out_name = args.out if args.out else f"{mt}.res"
    try:
        with open(out_name, 'w', encoding='utf-8') as f:
            f.write("model,K,avg_rhyme_penalty,pairs,lines\n")
            f.write(f"{mt},{K},{avg_pen:.6f},{n_pairs},{n_lines}\n")
        print(f"[Save] Wrote results to {out_name}")
    except Exception as e:
        print(f"Warning: failed to save results to {out_name}: {e}")


if __name__ == "__main__":
    main()
