# plot_reversed.py
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
        try:
            return float(s)
        except Exception:
            return np.nan
    return np.nan


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_reversed(results: dict):
    """
    Extracts reversed model data.
    Expected shape in JSON:
      results["reversed"] = {
        "perplexity": <optional number>,
        "K": {"1": <float>, "2": <float>, ...}
      }
    Returns: (ppl: float|np.nan, D: np.array shape (len(K_LIST),)) or (np.nan, None) if missing.
    """
    obj = results.get("reversed")
    if not isinstance(obj, dict):
        return np.nan, None
    ppl = to_float(obj.get("perplexity", np.nan))
    k_map = obj.get("K", {}) or {}
    D = np.array([to_float(k_map.get(str(k), np.nan)) for k in K_LIST], dtype=float)
    return ppl, D


def plot_reversed_rhyme_vs_K(ppl: float, D: np.ndarray):
    if D is None:
        return None
    mask = ~np.isnan(D)
    if mask.sum() == 0:
        return None

    plt.figure()
    plt.plot(np.array(K_LIST)[mask], D[mask], marker="o", label="Reversed")
    plt.xscale("log", base=2)
    plt.xticks(K_LIST, [str(k) for k in K_LIST])
    plt.xlabel("K")
    plt.ylabel("Разстояние")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # Optional: include perplexity in title if available
    if not np.isnan(ppl):
        plt.title(f"CharLSTM-Reverse (перплексия={ppl:.3f})")
    else:
        plt.title("CharLSTM-Reverse")
    # Single curve; legend is optional. Uncomment if desired.
    # plt.legend()

    out = OUT_DIR / "rhyme_vs_K_reversed.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def main():
    try:
        results = load_results(JSON_PATH)
    except Exception as e:
        print(f"Error: could not load JSON from {JSON_PATH}: {e}")
        return

    ppl, D = collect_reversed(results)
    if D is None:
        print("No 'reversed' data found in results; add it under key 'reversed'.")
        return

    out = plot_reversed_rhyme_vs_K(ppl, D)
    if out is None:
        print("No valid reversed data to plot (all values missing).")
        return

    print("Saved PDF:")
    print(" -", out)


if __name__ == "__main__":
    main()
