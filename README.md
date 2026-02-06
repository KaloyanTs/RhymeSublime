# Bulgarian Poetry Generators
## Project for the Computational Linguistics Master's program

Two primary generators and one reversed model with rhyme enforcement:
- Char LSTM with differentiable DP rhyme (char_dp)
- Token/BPE LSTM (token)
- Reversed per-line Char LSTM (RTL generation, LTR output), plus ABAB variant

Stress resolution (all models):
- First: dictionary [bg_dict_csv/single_stress.csv](bg_dict_csv/single_stress.csv)
- Fallbacks: stress predictor [stress.py](stress.py), then last vowel

## Install
```bash
pip install torch pandas matplotlib
```

## Train & Generate

Char DP
- Prepare: `python run.py prepare_char_dp`
- Train: `python run.py train_char_dp`
- Generate: `python run.py generate_char_dp "Автор" debug "{" 0.6`

Token/BPE
- Prepare: `python run.py prepare_token`
- Train: `python run.py train_token`
- Generate: `python run.py generate_token "Автор" debug "{" 0.6`

Reversed (RTL per line)
- Train: `python run.py train_reversed` (saves to parameters: modelFileName_reversed)
- Generate: `python run.py generate_reversed "Автор" debug "{\n" 0.6`
- ABAB: `python run.py generate_reversed_abab "Автор" debug "{\n" 0.6`

## Evaluation (Rhyme DP Penalty)
- Char/Token: `python accuracy.py --models char_dp token --N 1000 --seed 42 --debug`
- Reversed: `python accuracy_reversed.py --lines 1000 --debug`

## Plotting
- Multi-model by λ: `python plot_results.py` → saves PDFs under assets/
- Reversed only: `python plot_reversed.py` → saves assets/rhyme_vs_K_reversed.pdf

## Config
- Paths and hyperparameters: [parameters.py](parameters.py)
- Start/end/pad/unk tokens: `{`, `}`, `|`, `@`

Optional data prep:
- Dictionary sources and normalization: [get_dataset.py](get_dataset.py), [one_stress.py](one_stress.py)
- Beron JSONL ingestion: [add_beron.py](add_beron.py)
