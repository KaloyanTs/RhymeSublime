# Bulgarian Poetry Generation — Models and Pipeline

This project provides two Bulgarian poetry generators and a supporting stress dictionary pipeline. It includes:
- Character-level LSTM with differentiable DP rhyme loss
- Token-level LSTM over BPE tokens
- Stress dictionary creation and stress prediction utilities

## Project Structure
- Data prep and dictionary:
  - [get_dataset.py](c:\Users\NEW\Documents\Master\semester1\TII\project\get_dataset.py), [one_stress.py](c:\Users\NEW\Documents\Master\semester1\TII\project\one_stress.py), [bg_dict_csv/single_stress.csv](c:\Users\NEW\Documents\Master\semester1\TII\project\bg_dict_csv\single_stress.csv)
  - Optional Beron JSONL ingestion: [add_beron.py](c:\Users\NEW\Documents\Master\semester1\TII\project\add_beron.py)
- Models and training:
  - Char DP: [model_char_dp.py](c:\Users\NEW\Documents\Master\semester1\TII\project\model_char_dp.py), [train_char_dp.py](c:\Users\NEW\Documents\Master\semester1\TII\project\train_char_dp.py)
  - Char baseline: [model_char.py](c:\Users\NEW\Documents\Master\semester1\TII\project\model_char.py), [train_char.py](c:\Users\NEW\Documents\Master\semester1\TII\project\train_char.py)
  - Token/BPE: [model_token.py](c:\Users\NEW\Documents\Master\semester1\TII\project\model_token.py), [train_token.py](c:\Users\NEW\Documents\Master\semester1\TII\project\train_token.py)
- Generation:
  - Char DP: [generate_char_dp.py](c:\Users\NEW\Documents\Master\semester1\TII\project\generate_char_dp.py)
  - Char baseline: [generator_char.py](c:\Users\NEW\Documents\Master\semester1\TII\project\generator_char.py)
  - Token/BPE: [generate_token.py](c:\Users\NEW\Documents\Master\semester1\TII\project\generate_token.py)
- Orchestration and config:
  - CLI: [run.py](c:\Users\NEW\Documents\Master\semester1\TII\project\run.py)
  - Settings: [parameters.py](c:\Users\NEW\Documents\Master\semester1\TII\project\parameters.py)
  - Utilities: [utils.py](c:\Users\NEW\Documents\Master\semester1\TII\project\utils.py)

## Stress Dictionary Pipeline
- Download/source processing:
  - `python get_dataset.py` — fetches source dictionary data
  - `python one_stress.py` — filters/normalizes entries and writes [bg_dict_csv/single_stress.csv](c:\Users\NEW\Documents\Master\semester1\TII\project\bg_dict_csv\single_stress.csv)
- Optional ingestion of Beron JSONL:
  - `python add_beron.py beron_fast_full.jsonl` — appends stress-marked forms into the CSV (handles forms[] and string indices)
- Usage: Both generators consult the CSV first to resolve stress, then use the stress model; fallback is last vowel only if needed.

## Models Overview

### Token LSTM (BPE)
- Input: BPE tokens
- Generation: samples tokens per line, evaluates K candidates, picks best
- Rhyme evaluation: hard DP edit distance with phonetic substitution cost during selection
- Stress handling: tail starts at stress position via dictionary → model → vowel fallback
- Training loss: LM cross-entropy plus a rhyme term that encourages the last token before newline to match the previous line’s last token (weighted by `lambda_rhyme_token`). Richer rhyme is applied during generation via hard-DP scoring.

CLI
- Prepare: `python run.py prepare_token`
- Train: `python run.py train_token [checkpoint]`
- Perplexity: `python run.py perplexity_token`
- Generate: `python run.py generate_token AUTHOR [debug] "{seed" [temperature]`

### Char LSTM + Differentiable DP Rhyme (Char DP)
- Input: character tokens
- Training: adds differentiable DP rhyme loss; tail starts at stress (dictionary/model) rather than last vowel
- Generation: same tail extraction semantics as token generator; optional candidate sampling and repetition penalties
- Stress handling: dictionary → model → vowel fallback

CLI
- Prepare: `python run.py prepare_char_dp`
- Train: `python run.py train_char_dp [checkpoint]`
- Perplexity: `python run.py perplexity_char_dp`
- Generate: `python run.py generate_char_dp AUTHOR [debug] "{seed" [temperature]`

### Char LSTM Baseline (non-DP)
- Input: character tokens
- Training: standard LM loss
- Generation: simple character sampling

CLI
- Prepare: `python run.py prepare`
- Train: `python run.py train_char [checkpoint]`
- Perplexity: `python run.py perplexity_char`
- Generate: `python run.py generate_char AUTHOR "{seed" [temperature]`

## Pipeline Differences (Token vs Char DP)
- Tokens:
  - Unit: BPE tokens
  - Candidate selection: K sampled lines, scored by LL − rhyme − repetition
  - Tail extraction: last Cyrillic word of line; stress via dict/model
  - Debug: optional detailed distribution and scoring logs
  - Training rhyme: simple last-token cross-entropy (`_rhyme_loss_last`)
- Char DP:
  - Unit: individual characters
  - Training loss: LM + lambda × differentiable DP rhyme loss (soft edit distance)
  - Tail extraction: identical semantics to token generator; stress via dict/model
  - Generation: single-pass or candidate sampling; repetition penalty supported

## Quick Start
1) Install dependencies (example):
```bash
pip install torch pandas
```
2) Prepare data and train (Char DP example):
```bash
python run.py prepare_char_dp
python run.py train_char_dp
```
3) Generate:
```bash
python run.py generate_char_dp "Иван Вазов" debug
```

## Notes
- Special tokens: `{` start, `}` end, `@` unknown, `|` pad
- Configuration (paths, sizes, hyperparameters): [parameters.py](c:\Users\NEW\Documents\Master\semester1\TII\project\parameters.py)
- Stress model and utilities: [stress.py](c:\Users\NEW\Documents\Master\semester1\TII\project\stress.py), [stress_model.py](c:\Users\NEW\Documents\Master\semester1\TII\project\stress_model.py)
