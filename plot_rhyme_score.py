import sys
import os
import csv
import argparse

import matplotlib.pyplot as plt

modelMap = {
    "char": "LSTM",
    "token": "TokenLSTM",
    "char_dp": "CharLSTM-DP",
}

def read_res_file(path):
    """Read a .res CSV file into dict: model -> list of (K, avg_penalty)."""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Expect headers: model,K,avg_rhyme_penalty,pairs,lines
        for row in reader:
            try:
                model = row.get('model')
                K = int(row.get('K'))
                avg = float(row.get('avg_rhyme_penalty'))
            except Exception:
                # Skip malformed rows
                continue
            data.setdefault(model, []).append((K, avg))
    # Sort each model's points by K ascending
    for m in data:
        data[m].sort(key=lambda x: x[0])
    return data


def plot_lines(data, title, out_path):
    plt.figure(figsize=(8, 5))
    for model, points in data.items():
        if not points:
            continue
        xs = [k for k, _ in points]
        ys = [y for _, y in points]
        plt.plot(xs, ys, marker='o', linewidth=2, label=modelMap.get(model, model))
    plt.xlabel('K (брой семплирания при генериране)')
    plt.ylabel('Средно наказание за рима')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot rhyme score lines per model from a .res file.')
    parser.add_argument('res_file', help='Input .res file (CSV with model,K,avg_rhyme_penalty,...)')
    parser.add_argument('--out', default=os.path.join('assets', 'rhyme_scores.pdf'), help='Output image path (default assets/rhyme_scores.pdf)')
    args = parser.parse_args()

    res_file = args.res_file
    if not os.path.isfile(res_file):
        print(f"Input file not found: {res_file}")
        sys.exit(1)
    if not res_file.lower().endswith('.res'):
        print('Warning: input does not end with .res, attempting to read anyway...')

    data = read_res_file(res_file)
    if not data:
        print('No data found in file (check headers and content).')
        sys.exit(1)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out) or '.'
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    title = f"Rhyme Penalty vs K ({os.path.basename(res_file)})"
    plot_lines(data, title, args.out)


if __name__ == '__main__':
    main()
