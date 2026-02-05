# Bulgarian Poetry Models: Overview and Usage

This repository contains several language models for Bulgarian poetry generation and evaluation, along with a small stress dictionary pipeline.

## Dataset Pipeline (bg_dict_csv)
- Source: A raw Bulgarian dictionary dataset from Hugging Face.
- Creation: Run the downloader to create the directory and fetch CSV files into `bg_dict_csv/`:

```bash
python get_dataset.py
```

- Filtering: Keep only entries with exactly one stress (or add stress to single-vowel words) and write `single_stress.csv`:

```bash
python one_stress.py
```

- Result: `bg_dict_csv/single_stress.csv` is used by stress-related utilities during generation and evaluation.

## Models and Related Scripts

### Char LSTM + Rhyme (non-DP)
- Files: `model_char.py`, `train_char.py`, `generator_char.py`
- CLI:
  - Train: `python run.py train_char [optional_checkpoint]`
  - Perplexity: `python run.py perplexity_char`
  - Generate: `python run.py generate_char AUTHOR "{seed" [temperature]`

### Char LSTM + Differentiable DP Rhyme
- Files: `model_char_dp.py`, `train_char_dp.py`, `generate_char_dp.py`
- Description: Adds a differentiable dynamic programming rhyme loss based on soft edit distance with phonetic substitution costs.
- CLI:
  - Prepare: `python run.py prepare_char_dp`
  - Train: `python run.py train_char_dp [optional_checkpoint]`
  - Perplexity: `python run.py perplexity_char_dp`
  - Generate: `python run.py generate_char_dp AUTHOR "{seed" [temperature]`

### Token LSTM (BPE tokens)
- Files: `model_token.py`, `train_token.py`, `generate_token.py`
- CLI:
  - Prepare: `python run.py prepare_token`
  - Train: `python run.py train_token [optional_checkpoint]`
  - Perplexity: `python run.py perplexity_token`
  - Generate: `python run.py generate_token AUTHOR [debug] "{seed" [temperature]`

## Stress Utilities
- Files: `stress_model.py`, `stress.py`
- Purpose: Predict or annotate stress positions and optionally use the dictionary (`single_stress.csv`) during generation.

## Quick Setup

1. Install dependencies (example):
```bash
pip install torch pandas huggingface_hub
```

2. Prepare training corpus (characters):
```bash
python run.py prepare
```

3. Train a model (example - Char DP):
```bash
python run.py prepare_char_dp
python run.py train_char_dp
```

4. Generate text (example):
```bash
python run.py generate_char_dp AUTHOR "{Начало" 0.4
```

Notes:
- Special tokens: `{` start, `}` end, `@` unknown, `|` pad.
- Model, data file names, and hyperparameters are configured in `parameters.py`.

## Other Scripts
- [run.py](c:\Users\NEW\Documents\Master\semester1\TII\project\run.py): Unified CLI for data prep, training, perplexity, and generation across models.
- [parameters.py](c:\Users\NEW\Documents\Master\semester1\TII\project\parameters.py): Central configuration for file names, batch sizes, embedding sizes, and hyperparameters.
- [utils.py](c:\Users\NEW\Documents\Master\semester1\TII\project\utils.py): Corpus loading, author/symbol extraction, and tokenization (char/token/BPE).
- [print_corpus.py](c:\Users\NEW\Documents\Master\semester1\TII\project\print_corpus.py): Simple corpus inspection/printing utilities.
- [annotate.py](c:\Users\NEW\Documents\Master\semester1\TII\project\annotate.py): Helpers for annotating or debugging corpus lines (e.g., stress/rhyme markers).
- [closeness.py](c:\Users\NEW\Documents\Master\semester1\TII\project\closeness.py): Rhyme/phonetic closeness evaluation utilities for line endings.
- [hyphen.py](c:\Users\NEW\Documents\Master\semester1\TII\project\hyphen.py) and [hyph_bg_BG.dic](c:\Users\NEW\Documents\Master\semester1\TII\project\hyph_bg_BG.dic): Bulgarian hyphenation support; optional helpers for syllable/stress analysis.
- [dp.py](c:\Users\NEW\Documents\Master\semester1\TII\project\dp.py): Experimental dynamic programming utilities; prototypes used during development.
- [stress_model.py](c:\Users\NEW\Documents\Master\semester1\TII\project\stress_model.py) and [stress_model.pt](c:\Users\NEW\Documents\Master\semester1\TII\project\stress_model.pt): Stress prediction model and its weights used by generation/evaluation scripts.
