import argparse
import random
import pickle
import torch

from parameters import (
    device,
    char_emb_size_reversed,
    hid_size_reversed,
    lstm_layers_reversed,
    dropout_reversed,
    tokens2idFileName_reversed,
    auth2idFileName,
    modelFileName_reversed,
    defaultTemperature,
)

import model_reversed
import generate_reversed
from generate_reversed import extract_rime_ltr, _load_stress_map, _stress_index_from_stressed
from dp import rhyme_dp_penalty
import stress


def build_stress_dict(csv_path: str = "bg_dict_csv/single_stress.csv"):
    mp = _load_stress_map(csv_path)
    out = {}
    for word, stressed in mp.items():
        try:
            sidx = _stress_index_from_stressed(stressed, word)
            out[word] = sidx
        except Exception:
            pass
    return out


def load_model(tokens2id, auth2id):
    lm = model_reversed.CharAuthLSTM(
        vocab_size=len(tokens2id),
        auth2id=auth2id,
        emb_dim=char_emb_size_reversed,
        hidden_dim=hid_size_reversed,
        lstm_layers=lstm_layers_reversed,
        dropout=dropout_reversed,
        unk_token_idx=tokens2id.get('@', 0),
        line_end_token_idx=tokens2id.get('\n', None),
        tie_weights=False,
    ).to(device)
    try:
        state = torch.load(modelFileName_reversed, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            lm.load_state_dict(state['model'])
        else:
            lm.load_state_dict(state)
    except Exception as e:
        print('[AccuracyRTL] Warning: could not load', modelFileName_reversed, ':', e)
    return lm


def evaluate(lines_target: int = 1000, temperature: float = None, alpha_phonetic: float = 25.0, K: int = 1, debug: bool = False):
    # Load resources
    tokens2id = pickle.load(open(tokens2idFileName_reversed, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    id2auth = {v: k for k, v in auth2id.items()}
    lm = load_model(tokens2id, auth2id)

    stress_dict = build_stress_dict('bg_dict_csv/single_stress.csv')

    # Prepare author list (choose randomly for each sampling)
    authors = list(auth2id.keys())
    if not authors:
        raise RuntimeError('No authors available in auth2id')

    # Generation loop until reaching target lines
    lines_seen = 0
    pairs = 0
    total_penalty = 0.0

    while lines_seen < int(lines_target):
        # Pick a random author for this sampling iteration
        author = random.choice(authors)
        if debug:
            print('[AccuracyRTL] Sampling with author:', author)
        poem = generate_reversed.generateText_rtl_forced_rhyme(
            lm,
            tokens2id,
            author,
            '{\n',
            limit=5000,
            temperature=float(temperature) if temperature is not None else float(defaultTemperature),
            max_line_len=90,
            forbid_double_space=True,
            forbid_line_start_space=True,
            mark_stress=False,
            stress_predict_fn=stress.predict,
            stress_dict=stress_dict,
            debug=debug,
            alpha_phonetic=float(alpha_phonetic),
            K=int(K),
        )
        if debug:
            print("[AccuracyRTL] Poem output:\n" + poem)
        lines = [ln for ln in poem.split('\n') if ln is not None]
        if debug:
            print(f"[AccuracyRTL] Generated lines this pass: {len(lines)}")
        # Compute penalties for odd-even pairs: (1,2), (3,4), ...
        if lines and lines[0] == '':
            lines = lines[1:]
            if debug:
                print("[AccuracyRTL] Removed leading empty line, new line count:", len(lines))
        for i in range(0, len(lines) - 1, 2):
            l1 = lines[i]
            l2 = lines[i + 1]
            t1, _ = extract_rime_ltr(l1, stress_predict_fn=stress.predict, stress_dict=stress_dict)
            t2, _ = extract_rime_ltr(l2, stress_predict_fn=stress.predict, stress_dict=stress_dict)
            if t1 and t2:
                p = rhyme_dp_penalty(t1, t2)
                total_penalty += float(p)
                pairs += 1
                if debug:
                    print(f"[AccuracyRTL] Pair {pairs}: tail1={repr(t1)}, tail2={repr(t2)}, penalty={p:.4f}")
        lines_seen += len(lines)
        print(f"[AccuracyRTL] Progress: lines={lines_seen}, pairs={pairs}")

    avg_penalty = (total_penalty / max(1, pairs)) if pairs > 0 else 0.0
    print('[AccuracyRTL] Average rhyme penalty over', pairs, 'pairs:', avg_penalty)
    return avg_penalty


def main():
    parser = argparse.ArgumentParser(description='Evaluate reversed model rhyme quality')
    parser.add_argument('--lines', type=int, default=1000, help='Number of lines to generate and evaluate')
    parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature (default from parameters)')
    parser.add_argument('--alpha', type=float, default=25.0, help='Phonetic sensitivity alpha for forced tail')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug output')
    args = parser.parse_args()
    K_VALUES = [1, 2, 4, 8, 16, 32, 64]
    results = []
    for K in K_VALUES:
        avg = evaluate(lines_target=args.lines, temperature=args.temperature, alpha_phonetic=args.alpha, K=K, debug=args.debug)
        print(f"[SummaryRTL] K={K} avg_rhyme_penalty={avg:.4f}")
        results.append((K, avg))
    print("[FinalRTL] Average rhyme penalty per K:")
    for K, avg in results:
        print(f"  K={K}: {avg:.6f}")


if __name__ == '__main__':
    main()
