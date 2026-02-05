#############################################################################
### Домашно задание 3 — Annotate stress positions for words
#############################################################################

import os
import re
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont

# Vowel sets (Latin + Cyrillic)
VOWELS = set(list("aeiouyAEIOUY") + list("аеиоуъюяАЕИОУЪЮЯ"))

# Paths
CORPUS_DIR = 'corpusPoems'
WORDS_FILE = 'words'

WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)


def load_lexicon(path=WORDS_FILE):
    lex = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                base = t.replace('`','').lower()
                lex.add(base)
    return lex


def collect_corpus_words(corpus_path=CORPUS_DIR):
    freq = {}
    # Handle single-file corpus
    if os.path.isfile(corpus_path):
        for enc in ('utf-8', 'cp1251', 'latin-1'):
            try:
                with open(corpus_path, 'r', encoding=enc, errors='ignore') as f:
                    for line in f:
                        for tok in WORD_RE.findall(line):
                            w = tok.strip().lower()
                            if not w:
                                continue
                            w = w.strip(".,!?:;\"'()[]{}")
                            if not w or not any(ch.isalpha() for ch in w):
                                continue
                            freq[w] = freq.get(w, 0) + 1
                break
            except Exception:
                continue
        return freq
    # Otherwise, traverse directory
    for root, _, files in os.walk(corpus_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            file_read = False
            for enc in ('utf-8', 'cp1251', 'latin-1'):
                try:
                    with open(fpath, 'r', encoding=enc, errors='ignore') as f:
                        for line in f:
                            for tok in WORD_RE.findall(line):
                                w = tok.strip().lower()
                                if not w:
                                    continue
                                w = w.strip(".,!?:;\"'()[]{}")
                                if not w or not any(ch.isalpha() for ch in w):
                                    continue
                                freq[w] = freq.get(w, 0) + 1
                    file_read = True
                    break
                except Exception:
                    continue
            if not file_read:
                continue
    return freq


def save_stressed(word, index, path=WORDS_FILE):
    # Insert ` before the vowel at index
    stressed = word[:index] + '`' + word[index:]
    with open(path, 'a', encoding='utf-8') as f:
        f.write(stressed + "\n")


class AnnotatorApp:
    def __init__(self, master, words, freq):
        self.master = master
        self.words = words
        self.freq = freq
        self.idx = 0
        # Fonts
        self.font_large = tkfont.Font(family='Arial', size=16)
        self.font_button = tkfont.Font(family='Arial', size=16, weight='bold')

        self.label = tk.Label(master, text="", font=self.font_large)
        self.label.pack(pady=10)
        self.buttons_frame = tk.Frame(master)
        self.buttons_frame.pack(pady=10)
        # Stats under buttons
        self.stats_frame = tk.Frame(master)
        self.stats_frame.pack(pady=5)
        self.stats_label = tk.Label(self.stats_frame, text="", font=self.font_large, justify="left")
        self.stats_label.pack()
        self.ctrl_frame = tk.Frame(master)
        self.ctrl_frame.pack(pady=5)
        self.skip_btn = tk.Button(self.ctrl_frame, text="Skip", command=self.skip, font=self.font_button)
        self.skip_btn.pack(side=tk.LEFT, padx=5)
        self.back_btn = tk.Button(self.ctrl_frame, text="Back", command=self.back, font=self.font_button)
        self.back_btn.pack(side=tk.LEFT, padx=5)
        self.update_view()

    def update_view(self):
        for w in self.buttons_frame.winfo_children():
            w.destroy()
        # Auto-skip words without vowels
        while self.idx < len(self.words) and not any(ch in VOWELS for ch in self.words[self.idx]):
            self.idx += 1
        if self.idx >= len(self.words):
            messagebox.showinfo("Done", "All words processed.")
            self.master.quit()
            return
        word = self.words[self.idx]
        self.label.config(text=f"Select stress for: {word}  (freq: {self.freq.get(word, 0)})")
        # Render the word inline: labels for non-vowels, buttons for vowels
        word_frame = tk.Frame(self.buttons_frame)
        word_frame.pack(pady=10)
        for i, ch in enumerate(word):
            if ch in VOWELS:
                btn = tk.Button(word_frame, text=ch, command=lambda idx=i: self.choose(idx), font=self.font_button)
                btn.pack(side=tk.LEFT, padx=3, pady=3)
            else:
                lbl = tk.Label(word_frame, text=ch, font=self.font_large)
                lbl.pack(side=tk.LEFT, padx=1, pady=3)
        # Update stats: top 10 frequencies among remaining words
        freq_counts = {}
        for w in self.words[self.idx:]:
            f = self.freq.get(w, 0)
            freq_counts[f] = freq_counts.get(f, 0) + 1
        top_freqs = sorted(freq_counts.items(), key=lambda kv: kv[0], reverse=True)[:10]
        stats_lines = [f"{f}: {cnt} words" for f, cnt in top_freqs]
        self.stats_label.config(text="Top frequencies (freq: count)\n" + "\n".join(stats_lines))

    def choose(self, index):
        word = self.words[self.idx]
        save_stressed(word, index)
        self.idx += 1
        self.update_view()

    def skip(self):
        self.idx += 1
        self.update_view()

    def back(self):
        if self.idx > 0:
            self.idx -= 1
            self.update_view()


def main():
    lex = load_lexicon()
    freq = collect_corpus_words()
    # Filter out words already annotated
    unknown = [w for w in freq.keys() if w not in lex]
    # Auto-annotate words with exactly one vowel
    def single_vowel_index(word):
        idx = None
        count = 0
        for i, ch in enumerate(word):
            if ch in VOWELS:
                idx = i
                count += 1
                if count > 1:
                    return None
        return idx if count == 1 else None

    pending = []
    for w in unknown:
        idx = single_vowel_index(w)
        if idx is not None:
            save_stressed(w, idx)
        else:
            pending.append(w)
    unknown = pending
    # Sort remaining by decreasing frequency
    unknown.sort(key=lambda w: freq[w], reverse=True)
    root = tk.Tk()
    root.title("Stress Annotator")
    # Make UI bigger
    root.geometry('800x600')
    root.minsize(700, 500)
    app = AnnotatorApp(root, unknown, freq)
    root.mainloop()


if __name__ == '__main__':
    main()
