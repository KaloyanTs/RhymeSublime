import numpy as np
import matplotlib.pyplot as plt
import math

# --- Bulgarian lowercase letters (Cyrillic) ---
letters = list("абвгдежзийклмнопрстуфхцчшщъьюя")

# --- Фонетични признаци ---
# Всяка буква се представя чрез вектор от фонетични характеристики
# (ръчно дефинирани, лингвистично мотивирани)

# Размерности на признаците (двоични / категорийни, кодирани като цели числа):
# [гласна, звучност, място, начин, височина, заденост]
#
# място (за съгласни):
# 0 = няма, 1 = двуустна, 2 = устнозъбна, 3 = зъбна/алвеоларна,
# 4 = палатална, 5 = веларна
#
# начин (за съгласни):
# 0 = няма, 1 = взривна, 2 = фрикативна, 3 = африката,
# 4 = носова, 5 = сонорна, 6 = сложна
#
# височина (за гласни):
# 0 = няма, 1 = висока, 2 = средна, 3 = ниска
#
# заденост (за гласни):
# 0 = няма, 1 = предна, 2 = централна, 3 = задна

F = {}

# --- Vowels ---
for ch, h, b in [
    ("а",3,3), ("е",2,1), ("и",1,1), ("о",2,3),
    ("у",1,3), ("ъ",2,2), ("я",3,1), ("ю",1,3)
]:
    F[ch] = [1, 1, 0, 0, h, b]

# --- Consonants ---
for ch, v, p, m in [
    ("б",1,1,1), ("п",0,1,1),
    ("в",1,2,2), ("ф",0,2,2),
    ("д",1,3,1), ("т",0,3,1),
    ("з",1,3,2), ("с",0,3,2),
    ("ж",1,4,2), ("ш",0,4,2),
    ("г",1,5,1), ("к",0,5,1),
    ("м",1,1,4), ("н",1,3,4),
    ("л",1,3,5), ("р",1,3,5),
    ("й",1,4,5),
    ("х",0,5,2),
    ("ц",0,3,3), ("ч",0,4,3),
    ("щ",0,6,2), ("ь",0,0,0)
]:
    F[ch] = [0, v, p, m, 0, 0]

# --- Distance & similarity ---
def dist(a, b):
    fa, fb = F[a], F[b]
    d = 0.0
    for i in range(len(fa)):
        d += (fa[i] - fb[i]) ** 2
    d = math.sqrt(d)
    return d

def sim(a, b):
    return dist(a, b)

# --- Build matrix ---
N = len(letters)
M = np.zeros((N, N))

for i, a in enumerate(letters):
    for j, b in enumerate(letters):
        M[i, j] = sim(a, b)

# --- Plot ---
plt.figure(figsize=(10, 8))
im = plt.imshow(M, cmap="viridis")
plt.colorbar(im, label="Sound closeness")
plt.xticks(range(N), letters, rotation=90)
plt.yticks(range(N), letters)
plt.title("Bulgarian letter sound closeness (phonetic features)")

# Add text annotations with values
for i in range(N):
    for j in range(N):
        plt.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', 
                 color='white' if M[i, j] < 0.5 else 'black', fontsize=6)

plt.tight_layout()
plt.savefig("closeness_matrix.pdf", dpi=300)
plt.show()
