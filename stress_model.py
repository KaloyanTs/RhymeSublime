# stress_bilstm.py
# Train a char-level BiLSTM to predict Bulgarian word stress position
# using vislupus/bulgarian-dictionary-raw-data (name_stressed uses backticks ` after the stressed vowel).
#
# Install: pip install torch pandas numpy
#
# Train:
#   python stress_bilstm.py --csv words_combined.csv --out stress_model.pt --epochs 10
#
# Predict:
#   python stress_bilstm.py --load stress_model.pt --predict "планина"
#
import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


VOWELS_BG = set(list("аеиоуъюяѝАЕИОУЪЮЯЍ"))


def parse_stress_backtick(stressed: str) -> Optional[Tuple[str, int]]:
    """
    Input: string where a backtick ` appears AFTER the stressed character.
           Example: "ава`нпост" (stress on 'а' before the backtick)
    Output: (plain_word, stress_char_index_in_plain_word)
    Returns None if no stress mark or malformed.
    """
    if not isinstance(stressed, str) or "`" not in stressed:
        return None

    plain_chars = []
    stress_idx = None

    for ch in stressed:
        if ch == "`":
            # stress is on previous emitted char in plain_chars
            if len(plain_chars) == 0:
                return None
            stress_idx = len(plain_chars) - 1
        else:
            plain_chars.append(ch)

    if stress_idx is None:
        return None

    plain = "".join(plain_chars)
    if stress_idx < 0 or stress_idx >= len(plain):
        return None
    return plain, stress_idx


def has_exactly_one_stress(stressed: str) -> bool:
    return isinstance(stressed, str) and stressed.count("`") == 1


def build_vocab(words: List[str]) -> Dict[str, int]:
    chars = sorted(set("".join(words)))
    stoi = {"<pad>": 0, "<unk>": 1}
    for ch in chars:
        if ch not in stoi:
            stoi[ch] = len(stoi)
    return stoi


def word_to_ids(word: str, stoi: Dict[str, int]) -> List[int]:
    unk = stoi["<unk>"]
    return [stoi.get(ch, unk) for ch in word]


def make_vowel_mask(batch_words: List[str], max_len: int, device) -> torch.Tensor:
    """
    mask[b, t] = True if position t is a vowel char within the word length
    """
    mask = torch.zeros((len(batch_words), max_len), dtype=torch.bool, device=device)
    for b, w in enumerate(batch_words):
        for t, ch in enumerate(w):
            if ch in VOWELS_BG:
                mask[b, t] = True
    return mask


class StressDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Tuple[str, int]], stoi: Dict[str, int]):
        self.items = items
        self.stoi = stoi

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        word, stress_idx = self.items[idx]
        ids = word_to_ids(word, self.stoi)
        return word, torch.tensor(ids, dtype=torch.long), torch.tensor(len(ids), dtype=torch.long), torch.tensor(stress_idx, dtype=torch.long)


def collate_fn(batch):
    # batch: list of (word, ids, length, stress_idx)
    words = [x[0] for x in batch]
    lengths = torch.stack([x[2] for x in batch], dim=0)
    stress = torch.stack([x[3] for x in batch], dim=0)

    max_len = int(lengths.max().item())
    ids_padded = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (_, ids, _, _) in enumerate(batch):
        ids_padded[i, : ids.numel()] = ids

    return words, ids_padded, lengths, stress


class BiLSTMStress(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb,
            hidden_size=hid,
            num_layers=layers,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(2 * hid, 1)  # per-position logit

    def forward(self, x, lengths):
        # x: (B,L) ; lengths: (B,)
        e = self.emb(x)  # (B,L,E)
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # (B,L,2H)
        logits = self.proj(out).squeeze(-1)  # (B,L)
        return logits


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for words, x, lengths, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(x, lengths)  # (B,L)
        mask = make_vowel_mask(words, logits.size(1), device=device)
        # disallow non-vowels (and padded positions implicitly via mask)
        masked_logits = logits.masked_fill(~mask, -1e9)
        pred = masked_logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train(args):
    df = pd.read_csv(args.csv)

    if "name_stressed" not in df.columns:
        raise ValueError("CSV must contain a 'name_stressed' column.")

    rows = df["name_stressed"].dropna().astype(str).tolist()
    if args.only_one_stress:
        rows = [r for r in rows if has_exactly_one_stress(r)]

    items = []
    for r in rows:
        parsed = parse_stress_backtick(r)
        if parsed is None:
            continue
        w, sidx = parsed
        # stress should be on a vowel (skip weird entries)
        if sidx < 0 or sidx >= len(w) or w[sidx] not in VOWELS_BG:
            continue
        items.append((w, sidx))

    if len(items) < 1000:
        raise ValueError(f"Too few usable items ({len(items)}). Check that your CSV has many stressed forms.")

    random.seed(args.seed)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * (1.0 - args.val_frac))
    train_items = items[:n_train]
    val_items = items[n_train:]

    stoi = build_vocab([w for (w, _) in train_items])
    train_ds = StressDataset(train_items, stoi)
    val_ds = StressDataset(val_items, stoi)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BiLSTMStress(vocab_size=len(stoi), emb=args.emb, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        running = 0.0
        steps = 0

        for words, x, lengths, y in train_loader:
            print(f"  Batch {steps + 1}/{len(train_loader)}", end="\r")
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            logits = model(x, lengths)  # (B,L)

            # vowel-only classification over positions
            mask = make_vowel_mask(words, logits.size(1), device=device)
            masked_logits = logits.masked_fill(~mask, -1e9)

            loss = F.cross_entropy(masked_logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())
            steps += 1

        val_acc = accuracy(model, val_loader, device)
        avg_loss = running / max(1, steps)
        print(f"epoch {epoch}/{args.epochs}  train_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "stoi": stoi,
                    "config": {
                        "emb": args.emb,
                        "hid": args.hid,
                        "layers": args.layers,
                        "dropout": args.dropout,
                    },
                },
                args.out,
            )

    print(f"best val_acc={best_acc:.4f}  saved={args.out}")


@torch.no_grad()
def predict(args):
    ckpt = torch.load(args.load, map_location="cpu")
    stoi = ckpt["stoi"]
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BiLSTMStress(vocab_size=len(stoi), **cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    word = args.predict.strip()
    if not word:
        raise ValueError("Empty word.")

    ids = word_to_ids(word, stoi)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)

    logits = model(x, lengths)  # (1,L)
    mask = make_vowel_mask([word], logits.size(1), device=device)
    masked_logits = logits.masked_fill(~mask, -1e9)
    idx = int(masked_logits.argmax(dim=1).item())

    # print stressed form with backtick after stressed char (same style as dataset)
    stressed = word[: idx + 1] + "`" + word[idx + 1 :]
    print(stressed)
    print("stress_char_index =", idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, help="CSV file with name_stressed column (e.g. words_combined.csv)")
    p.add_argument("--out", type=str, default="stress_model.pt")
    p.add_argument("--only_one_stress", action="store_true", help="keep only rows with exactly one backtick stress")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--hid", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--load", type=str, help="Load a trained model checkpoint")
    p.add_argument("--predict", type=str, help="Predict stress for a single word")

    args = p.parse_args()

    if args.load and args.predict:
        predict(args)
        return

    if not args.csv:
        raise ValueError("Provide --csv to train, or --load + --predict to run inference.")

    # default to filtering single-stress rows (safer)
    if not args.only_one_stress:
        print("Tip: add --only_one_stress to avoid multi-stress compounds.")

    train(args)


if __name__ == "__main__":
    main()
