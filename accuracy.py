import sys
import random
import pickle
import argparse

import generator_char
import generate_token
import generate_char_dp
import model_char
import model_token
import model_char_dp

from parameters import (
    device,
    # char
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

def extract_tail_last_vowel(word: str):
    if not word:
        return "", 0
    sidx = 0
    for j in range(len(word) - 1, -1, -1):
        if word[j] in VOWELS_BG:
            sidx = j
            break
    tail = word[sidx:] if 0 <= sidx < len(word) else ""
    return tail, sidx


def load_env(model_type: str):
    mt = model_type.lower()
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    if mt == 'char':
        tokens2id = pickle.load(open(char2idFileName, 'rb'))
        lm = model_char.LSTMLanguageModelPack(
            char_emb_size, hid_size, auth2id, tokens2id,
            unkToken='@', padToken='|', endToken='}',
            lstm_layers=lstm_layers, dropout=dropout,
            k_rhyme=3, lambda_rhyme=0.1,
        ).to(device)
        lm.load(modelFileName, device)
        gen_fn = lambda auth, seed, K, temp: generator_char.generateText(
            lm, tokens2id, auth, seed, temperature=temp, K=K
        )
        return lm, tokens2id, auth2id, gen_fn
    elif mt == 'token':
        tokens2id = pickle.load(open(tokens2idFileName_token, 'rb'))
        lm = model_token.TokenLSTMLanguageModelPack(
            char_emb_size_token, hid_size_token, auth2id, tokens2id,
            unkToken='@', padToken='|', endToken='}',
            lstm_layers=lstm_layers_token, dropout=dropout_token,
            lambda_rhyme=0.1,
        ).to(device)
        lm.load(modelFileName_token, device)
        gen_fn = lambda auth, seed, K, temp: generate_token.generateText(
            lm, tokens2id, auth, seed, temperature=temp, K=K
        )
        return lm, tokens2id, auth2id, gen_fn
    elif mt == 'char_dp' or mt == 'char-dp':
        tokens2id = pickle.load(open(tokens2idFileName_dp, 'rb'))
        lm = model_char_dp.CharLSTMLanguageModelPack(
            char_emb_size_dp, hid_size_dp, auth2id, tokens2id,
            unkToken='@', padToken='|', endToken='}',
            lstm_layers=lstm_layers_dp, dropout=dropout_dp,
            lambda_rhyme=0.0,
        ).to(device)
        lm.load(modelFileName_dp, device)
        gen_fn = lambda auth, seed, K, temp: generate_char_dp.generateText(
            lm, tokens2id, auth, seed, temperature=temp, K=K
        )
        return lm, tokens2id, auth2id, gen_fn
    else:
        raise ValueError("Unsupported model_type. Use one of: char, token, char_dp")


def compute_pair_penalties(poem_text: str):
    """Return (penalties list, number of non-empty lines) for poem pairs (odd, even)."""
    lines = [ln for ln in poem_text.splitlines() if ln.strip() != ""]
    penalties = []
    for i in range(1, len(lines), 2):
        base = lines[i - 1]
        cand = lines[i]
        base_last = extract_last_word(base)
        cand_last = extract_last_word(cand)
        base_tail, _ = extract_tail_last_vowel(base_last)
        cand_tail, _ = extract_tail_last_vowel(cand_last)
        if base_tail and cand_tail:
            penalties.append(float(rhyme_penalty_str(base_tail, cand_tail)))
    return penalties, len(lines)


def run_for_model(mt: str, K: int, N_lines: int, seed: int):
    """Generate until total sampled lines >= N_lines; average over all pair penalties."""
    _, _, auth2id, gen_fn = load_env(mt)
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
            print("Total lines so far:", total_lines, end='\r', flush=True)
            poem = gen_fn(auth, start_seed, K, defaultTemperature)
            pen_list, lines_used = compute_pair_penalties(poem)
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
    args = parser.parse_args()

    K_VALUES = [1, 2, 4, 8, 16, 32, 64]

    if args.models:
        N = args.N if isinstance(args.N, int) else 1000
        results = []
        for mt in args.models:
            for K in K_VALUES:
                avg_pen, n_pairs, n_lines = run_for_model(mt, K, N, args.seed)
                print(f"model={mt} K={K} avg_rhyme_penalty={avg_pen:.4f} (pairs={n_pairs}, lines={n_lines})")
                results.append({
                    'model': mt,
                    "rhyme_loss": parameters.lambda_rhyme_token if mt == 'token' else (parameters.lambda_rhyme_dp if mt == 'char_dp' else parameters.lambda_rhyme),
                    'K': K,
                    'avg_rhyme_penalty': avg_pen,
                    'pairs': n_pairs,
                    'lines': n_lines,
                })
        # Save consolidated results
        out_name = f"{'_'.join(args.models)}.res"
        try:
            with open(out_name, 'w', encoding='utf-8') as f:
                f.write("model,K,avg_rhyme_penalty,pairs,lines\n")
                for r in results:
                    f.write(f"{r['model']},{r['K']},{r['avg_rhyme_penalty']:.6f},{r['pairs']},{r['lines']}\n")
            print(f"Saved results to {out_name}")
        except Exception as e:
            print(f"Warning: failed to save results to {out_name}: {e}")
        return

    # Fallback: single-model positional mode
    if not args.model or args.K is None:
        print("Usage: python accuracy.py --models <char token char_dp> [--N 1000] [--seed 42]\n       or: python accuracy.py <model_type> <K> [N]")
        sys.exit(1)

    mt = args.model
    K = int(args.K)
    N = int(args.N) if isinstance(args.N, int) else 1000
    avg_pen, n_pairs, n_lines = run_for_model(mt, K, N, args.seed)
    print(f"model={mt} K={K} avg_rhyme_penalty={avg_pen:.4f} (pairs={n_pairs}, lines={n_lines})")
    # Save single-model result
    out_name = f"{mt}.res"
    try:
        with open(out_name, 'w', encoding='utf-8') as f:
            f.write("model,K,avg_rhyme_penalty,pairs,lines\n")
            f.write(f"{mt},{K},{avg_pen:.6f},{n_pairs},{n_lines}\n")
        print(f"Saved results to {out_name}")
    except Exception as e:
        print(f"Warning: failed to save results to {out_name}: {e}")


if __name__ == "__main__":
    main()
