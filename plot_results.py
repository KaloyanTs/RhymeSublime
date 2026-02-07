import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


JSON_PATH = "results/database.json"
OUT_DIR = Path("assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [1, 2, 4, 8, 16, 32, 64]

def to_float(x):
    """Robust parse: accepts numbers, strings, None. Treats '?'/'' as missing (NaN)."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s in {"?", "", "nan", "NaN", "None"}:
            return np.nan
        return float(s)
    return np.nan


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_by_model(results: dict):
    """
    Returns:
      data[model][lambda_float] = {"ppl": float, "D": np.array shape (len(K_LIST),)}
    """
    data = {}
    for model_name, by_lambda in results.items():
        model_map = {}
        for lam_str, lam_obj in by_lambda.items():
            lam = float(lam_str)
            ppl = to_float(lam_obj.get("perplexity", np.nan))
            k_map = lam_obj.get("K", {}) or {}
            D = np.array([to_float(k_map.get(str(k), np.nan)) for k in K_LIST], dtype=float)
            model_map[lam] = {"ppl": ppl, "D": D}
        data[model_name] = model_map
    return data


def plot_rhyme_vs_K_for_lambda(data, lam: float):
    plt.figure()
    any_plotted = False

    for model_name, m in data.items():
        if lam not in m:
            continue
        D = m[lam]["D"]
        mask = ~np.isnan(D)
        if mask.sum() == 0:
            continue

        plt.plot(
            np.array(K_LIST)[mask],
            D[mask],
            marker="o",
            label=model_name,
        )
        any_plotted = True

    if not any_plotted:
        return None

    plt.xscale("log", base=2)
    plt.xticks(K_LIST, [str(k) for k in K_LIST])
    plt.xlabel("K")
    plt.ylabel("Разстояние")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(["CharLSTM-DP", "TokenLSTM"])

    out = OUT_DIR / f"rhyme_vs_K_lambda_{lam:.2f}.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def main():
    results = load_results(JSON_PATH)
    data = collect_by_model(results)

    all_lams = sorted({lam for m in data.values() for lam in m.keys()})
    saved = []
    for lam in all_lams:
        out = plot_rhyme_vs_K_for_lambda(data, lam)
        if out is not None:
            saved.append(out)
            
    print("Saved PDFs:")
    for p in saved:
        print(" -", p)


if __name__ == "__main__":
    main()
