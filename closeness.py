import numpy as np
import math
import matplotlib.pyplot as plt

letters = list("абвгдежзийклмнопрстуфхцчшщъьюя")

SEQ = {ch: [ch] for ch in letters}
SEQ["щ"] = ["ш", "т"]
SEQ["ю"] = ["й", "у"]
SEQ["я"] = ["й", "а"]
SEQ["ь"] = []

VOWELS = {"а","е","и","о","у","ъ"}

v_row = {
    "и":"предни", "е":"предни",
    "а":"задни", "ъ":"задни", "о":"задни", "у":"задни"
}
v_width = {
    "и":"тесни", "ъ":"тесни", "у":"тесни",
    "е":"широки", "а":"широки", "о":"широки"
}
v_lips = { 
    "о":"лабиални", "у":"лабиални",
    "и":"нелабиални", "е":"нелабиални", "а":"нелабиални", "ъ":"нелабиални"
}
c_place = {}
for ch in ["б","п","в","ф","м"]:
    c_place[ch] = "устнени/лабиални"
for ch in ["д","т","з","с","л","н","р","ц"]:
    c_place[ch] = "алвеолни"
for ch in ["ж","ш","ч","й"]:
    c_place[ch] = "меконебни/палатални"
for ch in ["г","к","х"]:
    c_place[ch] = "със задната част на езика"

c_manner = {}
for ch in ["б","п","д","т","г","к"]:
    c_manner[ch] = "преградни"
for ch in ["в","ф","з","с","ж","ш","х"]:
    c_manner[ch] = "проходни"
for ch in ["ц","ч"]:
    c_manner[ch] = "преградно-проходни"
for ch in ["м","н"]:
    c_manner[ch] = "носови/назални"
c_manner["л"] = "латерални"
c_manner["р"] = "вибрантни"
c_manner["й"] = "глайдове"

c_voice = {}
for ch in ["л","м","р","н","й"]:
    c_voice[ch] = "сонорни"
for ch in ["б","в","д","з","ж","г"]:
    c_voice[ch] = "звучни"
for ch in ["п","ф","т","с","ц","ч","ш","к","х"]:
    c_voice[ch] = "беззвучни"

W_V = {"редица":0.45, "широчина":0.45, "устни":0.10}
W_C = {"място":0.40, "начин":0.40, "звучност":0.20}

def segdist(a: str, b: str) -> float:
    """Разстояние между ДВА звука (фонеми), нормализирано 0..1."""
    if a == "" and b == "": return 0.0
    if a == "" or b == "": return 1.0

    a_is_v = a in VOWELS
    b_is_v = b in VOWELS
    if a_is_v != b_is_v:
        return 1.0  # гласна vs съгласна

    if a_is_v and b_is_v:
        d = 0.0
        d += W_V["редица"]    * (v_row[a]   != v_row[b])
        d += W_V["широчина"] * (v_width[a] != v_width[b])
        d += W_V["устни"]    * (v_lips[a]  != v_lips[b])
        return float(d)

    d = 0.0
    d += W_C["място"] * (c_place.get(a,"") != c_place.get(b,""))
    d += W_C["начин"] * (c_manner.get(a,"") != c_manner.get(b,""))

    va, vb = c_voice.get(a,""), c_voice.get(b,"")
    if va == vb:
        dv = 0.0
    else:
        if "сонорни" in (va, vb):
            dv = 1.0
        else:
            dv = 1.0  # звучни vs беззвучни
    d += W_C["звучност"] * dv
    return float(d)

def seqdist(A, B) -> float:
    """Levenshtein разстояние между ДВЕ последователности от фонеми, 0..1."""
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0
    # DP
    dp = np.zeros((n+1, m+1), dtype=float)
    for i in range(1, n+1):
        dp[i, 0] = dp[i-1, 0] + 1.0  # изтриване
    for j in range(1, m+1):
        dp[0, j] = dp[0, j-1] + 1.0  # вмъкване
    for i in range(1, n+1):
        for j in range(1, m+1):
            sub = dp[i-1, j-1] + segdist(A[i-1], B[j-1])
            ins = dp[i, j-1] + 1.0
            dele = dp[i-1, j] + 1.0
            dp[i, j] = min(sub, ins, dele)
    return float(dp[n, m] / max(n, m))  # нормализация

def dist(a: str, b: str) -> float:
    return seqdist(SEQ[a], SEQ[b])

def sim(a: str, b: str) -> float:
    return 5*dist(a, b)

if __name__ == "__main__":
    N = len(letters)
    M = np.zeros((N, N))

    for i, a in enumerate(letters):
        for j, b in enumerate(letters):
            M[i, j] = sim(a, b)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(M, cmap="viridis")
    plt.colorbar(im, label="Разстояние между звуците (фонетична разлика)")
    plt.xticks(range(N), letters, rotation=90)
    plt.yticks(range(N), letters)
    plt.title("Разстояние между букви (определени от фонетични признаци)")

    for i in range(N):
        for j in range(N):
            plt.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', 
                     color='white' if M[i, j] < 0.5 else 'black', fontsize=6)

    plt.tight_layout()
    plt.savefig("closeness_matrix.pdf", dpi=300)
    plt.show()
