import math
import argparse
import sys

# Phonetic feature vectors for Bulgarian lowercase letters (Cyrillic)
# Feature dims: [vowel(0/1), voicing(0/1), place(0..6), manner(0..6), height(0..3), backness(0..3)]
PHONETIC_FEATURES = {}

# Vowels: (char, height, backness)
for ch, h, b in [
    ("а", 3, 3), ("е", 2, 1), ("и", 1, 1), ("о", 2, 3),
    ("у", 1, 3), ("ъ", 2, 2), ("я", 3, 1), ("ю", 1, 3),
]:
    PHONETIC_FEATURES[ch] = [1, 1, 0, 0, h, b]

# Consonants: (char, voicing, place, manner)
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


def phonetic_dist(a: str, b: str) -> float:
    """Euclidean distance between phonetic feature vectors for Bulgarian letters.
    Falls back to 1.0 if either char is unsupported."""
    if a == b:
        return 0.0
    a_l = a.lower()
    b_l = b.lower()
    fa = PHONETIC_FEATURES.get(a_l)
    fb = PHONETIC_FEATURES.get(b_l)
    if fa is None or fb is None:
        return 1.0
    return math.sqrt(sum((fa[i] - fb[i]) ** 2 for i in range(len(fa))))


def rhyme_dp_penalty(base_tail: str, cand_tail: str) -> float:
    """Compute DP rhyme penalty using:
    - Insertion/deletion cost: 10.0 per character
    - Substitution cost: phonetic_dist between characters
    - Extra penalty: +10.0 if first letters differ (case-insensitive)
    Returns a positive penalty (lower is better)."""
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
                dp[i - 1][j] + 10.0,               # deletion
                dp[i][j - 1] + 10.0,               # insertion
                dp[i - 1][j - 1] + phonetic_dist(ai, bj),  # substitution
            )
    dist = dp[n][m]
    if base_tail and cand_tail and base_tail[0].lower() != cand_tail[0].lower():
        dist += 10.0
    return dist


def main():
    
    w1 = sys.argv[1]
    w2 = sys.argv[2]
    penalty = rhyme_dp_penalty(w1, w2)
    print(f"{penalty:.6f}")


if __name__ == "__main__":
    main()
