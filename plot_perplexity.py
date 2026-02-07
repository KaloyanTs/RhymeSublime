import json

import matplotlib.pyplot as plt

with open('results/database.json', 'r') as f:
    data = json.load(f)

lambda_keys = data["char_dp"].keys()
lambdas = [float(key) for key in data['char_dp'].keys()]
char_dp_perplexity = [data['char_dp'][key]['perplexity'] for key in sorted(data['char_dp'].keys())]
token_perplexity = [data['token'][key]['perplexity'] for key in sorted(data['token'].keys())]

lambdas_sorted = sorted(lambdas)
char_dp_perplexity_sorted = [data['char_dp'][l]['perplexity'] for l in lambda_keys]
token_perplexity_sorted = [data['token'][l]['perplexity'] for l in lambda_keys]

import json
import os

import matplotlib.pyplot as plt

with open('results/database.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

char_keys_sorted = sorted(data['char_dp'].keys(), key=lambda s: float(s))
token_keys_sorted = sorted(data['token'].keys(), key=lambda s: float(s))

char_lambdas = [float(k) for k in char_keys_sorted]
token_lambdas = [float(k) for k in token_keys_sorted]

char_perplexity = [data['char_dp'][k]['perplexity'] for k in char_keys_sorted]
token_perplexity = [data['token'][k]['perplexity'] for k in token_keys_sorted]

os.makedirs('assets', exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(char_lambdas, char_perplexity, marker='o', label='CharLSTM-DP', linewidth=2)
plt.xlabel('Фокус върху римата', fontsize=12)
plt.ylabel('Перплексия', fontsize=12)
plt.legend(["CharLSTM-DP"], fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/perplexity_char_dp.pdf', dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(token_lambdas, token_perplexity, marker='s', label='TokenLSTM', linewidth=2, color='orange')
plt.xlabel('Фокус върху римата', fontsize=12)
plt.ylabel('Перплексия', fontsize=12)
plt.legend(["TokenLSTM"], fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/perplexity_token.pdf', dpi=300)