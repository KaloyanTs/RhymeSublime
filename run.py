import sys
import model_char_dp
import model_token
import train_token
import train_char_dp
import generate_token
import generate_char_dp
import model_reversed
import train_reversed
import generate_reversed
import generate_reversed_abab
import stress
import torch
import pandas as pd

import utils
import pickle

VOWELS = set("aeiouyAEIOUY" + "аеиоуъюяАЕИОУЪЮЯ" + "ѝЍ")

def load_stress_dict(csv_path):
    """Load stress dictionary from CSV with format: word,stress_index"""
    try:
        df = pd.read_csv(csv_path)
        # Assume CSV has columns: word (plain), stress_index (int)
        # Or if it has 'name' and 'name_stressed' like the training data
        stress_dict = {}
        print(f"Loading stress dictionary from {csv_path}...")
        if 'name_stressed' in df.columns and 'name' in df.columns:
            print("Using 'name' and 'name_stressed' columns")
            for _, row in df.iterrows():
                word = str(row['name']).strip()
                idx = row['name_stressed'].find("`")
                # print(f"  Processing word: {word}, stressed form: {row['name_stressed']}, stress index: {idx}")
                if word and idx >= 1:
                    stress_dict[word] = idx  - 1
                    # print(f"    Added to dictionary: {word} -> {word[:idx-1]} + '`' + {word[idx-1:]}")
        elif 'name_stressed' in df.columns:
            # Parse backtick format: "ава`нпост" -> ("аванпост", 3)
            from stress_model import parse_stress_backtick
            for _, row in df.iterrows():
                stressed = str(row['name_stressed'])
                parsed = parse_stress_backtick(stressed)
                if parsed:
                    word, idx = parsed
                    stress_dict[word] = idx
        # print(f"Loaded {len(stress_dict)} words from stress dictionary")
        return stress_dict
    except Exception as e:
        print(f"Warning: Could not load stress dictionary from {csv_path}: {e}")
        return {}

from parameters import *

startChar = '{'
endChar = '}'
unkChar = '@'
padChar = '|'

if len(sys.argv)>1 and sys.argv[1] == 'prepare_token':
    print('Preparing data for token LSTM model...')
    testCorpus, trainCorpus, tokens2id, auth2id = utils.prepareData(
        corpusFileName_token, startChar, endChar, unkChar, padChar,
        maxPoemLength=maxPoemLength_token, tokenization="bpe", bpe_merges=merges_token
    )
    pickle.dump(testCorpus, open(testDataFileName_token, 'wb'))
    pickle.dump(trainCorpus, open(trainDataFileName_token, 'wb'))
    pickle.dump(tokens2id, open(tokens2idFileName_token, 'wb'))
    pickle.dump(auth2id, open(auth2idFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and sys.argv[1] == 'prepare_char_dp':
    print('Preparing data for char-DP model...')
    testCorpus, trainCorpus, tokens2id, auth2id = utils.prepareData(
        corpusFileName_dp, startChar, endChar, unkChar, padChar,
        maxPoemLength=maxPoemLength_dp
    )
    pickle.dump(testCorpus, open(testDataFileName_dp, 'wb'))
    pickle.dump(trainCorpus, open(trainDataFileName_dp, 'wb'))
    pickle.dump(tokens2id, open(tokens2idFileName_dp, 'wb'))
    pickle.dump(auth2id, open(auth2idFileName, 'wb'))
    print('Data prepared.')
    
if len(sys.argv)>1 and sys.argv[1] == 'train_token':
    testCorpus = pickle.load(open(testDataFileName_token, 'rb'))
    trainCorpus = pickle.load(open(trainDataFileName_token, 'rb'))
    tokens2id = pickle.load(open(tokens2idFileName_token, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))

    lm = model_token.TokenLSTMLanguageModelPack(
        char_emb_size_token, hid_size_token, auth2id, tokens2id,
        unkChar, padChar, endChar, lstm_layers=lstm_layers_token,
        dropout=dropout_token, lambda_rhyme=lambda_rhyme_token
    ).to(device)
    if len(sys.argv)>2: lm.load(sys.argv[2])

    optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
    train_token.trainModel(trainCorpus, testCorpus, lm, optimizer, epochs, batchSize_token)
    lm.save(modelFileName_token)
    print('Model perplexity: ', train_token.perplexity(lm, testCorpus, batchSize_token))

if len(sys.argv)>1 and sys.argv[1] == 'train_char_dp':
    testCorpus = pickle.load(open(testDataFileName_dp, 'rb'))
    trainCorpus = pickle.load(open(trainDataFileName_dp, 'rb'))
    tokens2id = pickle.load(open(tokens2idFileName_dp, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))

    lm = model_char_dp.CharLSTMLanguageModelPack(
        char_emb_size_dp, hid_size_dp, auth2id, tokens2id,
        unkChar, padChar, endChar, lstm_layers=lstm_layers_dp,
        dropout=dropout_dp, lambda_rhyme=lambda_rhyme_dp
    ).to(device)
    if len(sys.argv)>2:
        lm.load(sys.argv[2], device)

    optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
    train_char_dp.trainModel(trainCorpus, testCorpus, lm, optimizer, epochs, batchSize_dp)
    lm.save(modelFileName_dp)
    print('Model perplexity: ', train_char_dp.perplexity(lm, testCorpus, batchSize_dp))
    
if len(sys.argv)>1 and sys.argv[1] == 'train_reversed':
    # Build raw-text corpora so RTL can be applied BEFORE tokenization
    with open(corpusFileName_reversed, 'r', encoding='utf-8') as f:
        poems = f.read().split(utils.corpusSplitString)
    corpus = []
    for s in poems:
        if len(s) > 0:
            n = s.find('\n')
            aut = s[:n]
            poem = s[n+1:]
            text = startChar + poem[:maxPoemLength_reversed] + endChar
            corpus.append((aut, text))
    testCorpus, trainCorpus = utils.splitSentCorpus(corpus, testFraction=0.01)

    tokens2id = pickle.load(open(tokens2idFileName_reversed, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))

    lm = model_reversed.CharAuthLSTM(
        vocab_size=len(tokens2id),
        auth2id=auth2id,
        emb_dim=char_emb_size_reversed,
        hidden_dim=hid_size_reversed,
        lstm_layers=lstm_layers_reversed,
        dropout=dropout_reversed,
        unk_token_idx=tokens2id.get(unkChar, 0),
        line_end_token_idx=tokens2id.get('\n', None),
        tie_weights=False,
    ).to(device)
    if len(sys.argv)>2:
        # Optional: load a checkpoint; support both raw state_dict and {'model': state_dict}
        try:
            lm_state = torch.load(sys.argv[2], map_location=device)
            if isinstance(lm_state, dict) and 'model' in lm_state:
                lm.load_state_dict(lm_state['model'])
            else:
                lm.load_state_dict(lm_state)
            print('[Reversed] Loaded checkpoint', sys.argv[2])
        except Exception as e:
            print('[Reversed] Warning: could not load', sys.argv[2], ':', e)

    optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
    train_reversed.trainModel_rtl(
        trainCorpus, testCorpus, lm, optimizer, epochs, batchSize_reversed,
        tokens2id=tokens2id, rtl=True, use_amp=True
    )
    torch.save(lm.state_dict(), modelFileName_reversed)
    print('[Reversed] Saved model to', modelFileName_reversed)

if len(sys.argv)>1 and sys.argv[1] == 'perplexity_reversed':
    # Build raw-text test corpus to match RTL preprocessing
    with open(corpusFileName_reversed, 'r', encoding='utf-8') as f:
        poems = f.read().split(utils.corpusSplitString)
    corpus = []
    for s in poems:
        if len(s) > 0:
            n = s.find('\n')
            aut = s[:n]
            poem = s[n+1:]
            text = startChar + poem[:maxPoemLength_reversed] + endChar
            corpus.append((aut, text))
    testCorpus, _trainCorpus = utils.splitSentCorpus(corpus, testFraction=0.01)

    tokens2id = pickle.load(open(tokens2idFileName_reversed, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))

    lm = model_reversed.CharAuthLSTM(
        vocab_size=len(tokens2id),
        auth2id=auth2id,
        emb_dim=char_emb_size_reversed,
        hidden_dim=hid_size_reversed,
        lstm_layers=lstm_layers_reversed,
        dropout=dropout_reversed,
        unk_token_idx=tokens2id.get(unkChar, 0),
        line_end_token_idx=tokens2id.get('\n', None),
        tie_weights=False,
    ).to(device)
    # Load saved reversed model
    try:
        lm.load_state_dict(torch.load(modelFileName_reversed, map_location=device))
    except Exception as e:
        print('[Reversed] Warning: could not load', modelFileName_reversed, ':', e)

    p = train_reversed.perplexity_rtl(
        lm, testCorpus, batchSize_reversed, tokens2id=tokens2id, rtl=True, use_amp=True
    )
    print('[Reversed] Perplexity:', p)
    

if len(sys.argv)>1 and sys.argv[1] == 'perplexity_token':
    testCorpus = pickle.load(open(testDataFileName_token, 'rb'))
    tokens2id = pickle.load(open(tokens2idFileName_token, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_token.TokenLSTMLanguageModelPack(
        char_emb_size_token, hid_size_token, auth2id, tokens2id,
        unkChar, padChar, endChar, lstm_layers=lstm_layers_token,
        dropout=dropout_token, lambda_rhyme=lambda_rhyme_token
    ).to(device)
    lm.load(modelFileName_token, device)
    print('Model perplexity: ', train_token.perplexity(lm, testCorpus, batchSize_token))

if len(sys.argv)>1 and sys.argv[1] == 'perplexity_char_dp':
    testCorpus = pickle.load(open(testDataFileName_dp, 'rb'))
    tokens2id = pickle.load(open(tokens2idFileName_dp, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_char_dp.CharLSTMLanguageModelPack(
        char_emb_size_dp, hid_size_dp, auth2id, tokens2id,
        unkChar, padChar, endChar, lstm_layers=lstm_layers_dp,
        dropout=dropout_dp, lambda_rhyme=lambda_rhyme_dp
    ).to(device)
    lm.load(modelFileName_dp, device)
    print('Model perplexity: ', train_char_dp.perplexity(lm, testCorpus, batchSize_dp))
    
if len(sys.argv)>1 and sys.argv[1] == 'generate_token':
    print(sys.argv)
    
    if len(sys.argv)>2:
        auth = sys.argv[2]
    else:
        print('Usage: python run.py generate_token author K [debug [seed [temperature]]]')
        sys.exit(1)

    if len(sys.argv)>3:
        try:
            K = int(sys.argv[3])
        except Exception:
            print('K must be an integer')
            sys.exit(1)
    else:
        print('Usage: python run.py generate_token author K [debug [seed [temperature]]]')
        sys.exit(1)

    debug = False
    if len(sys.argv)>4:
        debug = True

    if len(sys.argv)>5:
        seed = sys.argv[5]
    else:
        seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>6:
        temperature = float(sys.argv[6])
    else:
        temperature = defaultTemperature

    tokens2id = pickle.load(open(tokens2idFileName_token, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_token.TokenLSTMLanguageModelPack(
        char_emb_size_token, hid_size_token, auth2id, tokens2id,
        unkChar, padChar, endChar, lstm_layers=lstm_layers_token,
        dropout=dropout_token, lambda_rhyme=lambda_rhyme_token
    ).to(device)
    lm.load(modelFileName_token, device)

    stress_dict = load_stress_dict('bg_dict_csv/single_stress.csv')

    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generate_token.generateText(lm, tokens2id, auth, seed, temperature=temperature, K=K, stress_predict=stress.predict, stress_dict=stress_dict, debug=debug))

if len(sys.argv)>1 and sys.argv[1] == 'generate_char_dp':
    print(sys.argv)
    
    if len(sys.argv)>2:
        auth = sys.argv[2]
    else:
        print('Usage: python run.py generate_char_dp author K [debug [seed [temperature]]]')
        sys.exit(1)

    if len(sys.argv)>3:
        try:
            K = int(sys.argv[3])
        except Exception:
            print('K must be an integer')
            sys.exit(1)
    else:
        print('Usage: python run.py generate_char_dp author K [debug [seed [temperature]]]')
        sys.exit(1)

    debug = False
    if len(sys.argv)>4:
        debug = True

    if len(sys.argv)>5:
        seed = sys.argv[5]
    else:
        seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>6:
        temperature = float(sys.argv[6])
    else:
        temperature = defaultTemperature

    tokens2id = pickle.load(open(tokens2idFileName_dp, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_char_dp.CharLSTMLanguageModelPack(
        char_emb_size_dp, hid_size_dp, auth2id, tokens2id,
        unkChar, padChar, endChar, lstm_layers=lstm_layers_dp,
        dropout=dropout_dp, lambda_rhyme=lambda_rhyme_dp
    ).to(device)
    lm.load(modelFileName_dp, device)
    
    stress_dict = load_stress_dict('bg_dict_csv/single_stress.csv')

    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generate_char_dp.generateText(lm, tokens2id, auth, seed, temperature=temperature, K=K, debug=debug, stress_predict=stress.predict, stress_dict=stress_dict))

if len(sys.argv)>1 and sys.argv[1] == 'generate_reversed':
    print(sys.argv)

    if len(sys.argv)>2:
        auth = sys.argv[2]
    else:
        print('Usage: python run.py generate_reversed author K [debug [seed [temperature]]]')
        sys.exit(1)

    if len(sys.argv)>3:
        try:
            K = int(sys.argv[3])
        except Exception:
            print('K must be an integer')
            sys.exit(1)
    else:
        print('Usage: python run.py generate_reversed author K [debug [seed [temperature]]]')
        sys.exit(1)

    debug = False
    if len(sys.argv)>4:
        debug = True

    if len(sys.argv)>5:
        seed = sys.argv[5]
    else:
        seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>6:
        temperature = float(sys.argv[6])
    else:
        temperature = defaultTemperature

    tokens2id = pickle.load(open(tokens2idFileName_reversed, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_reversed.CharAuthLSTM(
        vocab_size=len(tokens2id),
        auth2id=auth2id,
        emb_dim=char_emb_size_reversed,
        hidden_dim=hid_size_reversed,
        lstm_layers=lstm_layers_reversed,
        dropout=dropout_reversed,
        unk_token_idx=tokens2id.get(unkChar, 0),
        line_end_token_idx=tokens2id.get('\n', None),
        tie_weights=False,
    ).to(device)
    try:
        lm_state = torch.load(modelFileName_reversed, map_location=device)
        # Support both raw state_dict and {'model': state_dict}
        if isinstance(lm_state, dict) and 'model' in lm_state:
            lm.load_state_dict(lm_state['model'])
        else:
            lm.load_state_dict(lm_state)
    except Exception as e:
        print('[Reversed] Warning: could not load', modelFileName_reversed, ':', e)

    stress_dict = load_stress_dict('bg_dict_csv/single_stress.csv')
    
    print(f"Generating reversed poem for author '{auth}' with seed '{seed}' and temperature {temperature}...")

    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generate_reversed.generateText_rtl_forced_rhyme(
        lm, tokens2id, auth, seed,
        temperature=temperature,
        stress_predict_fn=stress.predict,
        stress_dict=stress_dict,
        debug=debug,
        K=K
    ))

if len(sys.argv)>1 and sys.argv[1] == 'generate_reversed_abab':
    print(sys.argv)

    if len(sys.argv)>2:
        auth = sys.argv[2]
    else:
        print('Usage: python run.py generate_reversed_abab author K [debug [seed [temperature]]]')
        sys.exit(1)

    if len(sys.argv)>3:
        try:
            K = int(sys.argv[3])
        except Exception:
            print('K must be an integer')
            sys.exit(1)
    else:
        print('Usage: python run.py generate_reversed_abab author K [debug [seed [temperature]]]')
        sys.exit(1)

    debug = False
    if len(sys.argv)>4:
        debug = True

    if len(sys.argv)>5:
        seed = sys.argv[5]
    else:
        seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>6:
        temperature = float(sys.argv[6])
    else:
        temperature = defaultTemperature

    tokens2id = pickle.load(open(tokens2idFileName_reversed, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_reversed.CharAuthLSTM(
        vocab_size=len(tokens2id),
        auth2id=auth2id,
        emb_dim=char_emb_size_reversed,
        hidden_dim=hid_size_reversed,
        lstm_layers=lstm_layers_reversed,
        dropout=dropout_reversed,
        unk_token_idx=tokens2id.get(unkChar, 0),
        line_end_token_idx=tokens2id.get('\n', None),
        tie_weights=False,
    ).to(device)
    try:
        lm_state = torch.load(modelFileName_reversed, map_location=device)
        if isinstance(lm_state, dict) and 'model' in lm_state:
            lm.load_state_dict(lm_state['model'])
        else:
            lm.load_state_dict(lm_state)
    except Exception as e:
        print('[ReversedABAB] Warning: could not load', modelFileName_reversed, ':', e)

    stress_dict = load_stress_dict('bg_dict_csv/single_stress.csv')
    print(f"Generating reversed ABAB poem for author '{auth}' with seed '{seed}' and temperature {temperature}...")

    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generate_reversed_abab.generateText_rtl_abab(
        lm, tokens2id, auth, seed,
        temperature=temperature,
        stress_predict_fn=stress.predict,
        stress_dict=stress_dict,
        debug=debug,
        K=K
    ))
    