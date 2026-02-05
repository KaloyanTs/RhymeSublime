import sys
import generator_char
import model_char
import model_char_dp
import model_transformer
import model_token
import train_transformer
import train_token
import train_char_dp
import generator_transformer
import generate_token
import generate_char_dp
import stress
import torch
import pandas as pd

import train_char
import utils
import generator
import train
import model
import pickle

import re

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

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    testCorpus, trainCorpus, char2id, auth2id =  utils.prepareData(corpusFileName, startChar, endChar, unkChar, padChar, maxPoemLength=maxPoemLength)
    pickle.dump(testCorpus, open(testDataFileName, 'wb'))
    pickle.dump(trainCorpus, open(trainDataFileName, 'wb'))
    pickle.dump(char2id, open(char2idFileName, 'wb'))
    pickle.dump(auth2id, open(auth2idFileName, 'wb'))
    print('Data prepared.')

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

if len(sys.argv)>1 and sys.argv[1] == 'train':
    testCorpus = pickle.load(open(testDataFileName, 'rb'))
    trainCorpus = pickle.load(open(trainDataFileName, 'rb'))
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))

    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
    if len(sys.argv)>2: lm.load(sys.argv[2])

    optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
    train.trainModel(trainCorpus, testCorpus, lm, optimizer, epochs, batchSize)
    lm.save(modelFileName)
    print('Model perplexity: ',train.perplexity(lm, testCorpus, batchSize))

if len(sys.argv)>1 and sys.argv[1] == 'perplexity':
    testCorpus = pickle.load(open(testDataFileName, 'rb'))
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
    lm.load(modelFileName,device)
    print('Model perplexity: ',train.perplexity(lm, testCorpus, batchSize))

if len(sys.argv)>1 and sys.argv[1] == 'generate':
    if len(sys.argv)>2: auth = sys.argv[2]
    else:
        print('Usage: python run.py generate author [seed [temperature]]')
    if len(sys.argv)>3: seed = sys.argv[3]
    else: seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>4: temperature = float(sys.argv[4])
    else: temperature = defaultTemperature
 
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
    lm.load(modelFileName,device)
    
    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generator.generateText(lm, char2id, auth, seed, temperature=temperature))
    
if len(sys.argv)>1 and sys.argv[1] == 'train_char':
    testCorpus = pickle.load(open(testDataFileName, 'rb'))
    trainCorpus = pickle.load(open(trainDataFileName, 'rb'))
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))

    lm = model_char.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout, k_rhyme=k_rhyme, lambda_rhyme=lambda_rhyme).to(device)
    if len(sys.argv)>2: lm.load(sys.argv[2])

    optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
    train_char.trainModel(trainCorpus, testCorpus, lm, optimizer, epochs, batchSize)
    lm.save(modelFileName)
    print('Model perplexity: ',train_char.perplexity(lm, testCorpus, batchSize))

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
    
if len(sys.argv)>1 and sys.argv[1] == 'perplexity_char':
    testCorpus = pickle.load(open(testDataFileName, 'rb'))
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_char.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout, k_rhyme=k_rhyme, lambda_rhyme=lambda_rhyme).to(device)
    lm.load(modelFileName,device)
    print('Model perplexity: ',train_char.perplexity(lm, testCorpus, batchSize))

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
    
if len(sys.argv)>1 and sys.argv[1] == 'generate_char':
    if len(sys.argv)>2: auth = sys.argv[2]
    else:
        print('Usage: python run.py generate_char author [seed [temperature]]')
    if len(sys.argv)>3: seed = sys.argv[3]
    else: seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>4: temperature = float(sys.argv[4])
    else: temperature = defaultTemperature
 
    char2id = pickle.load(open(char2idFileName, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_char.LSTMLanguageModelPack(char_emb_size, hid_size, auth2id, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout, k_rhyme=k_rhyme, lambda_rhyme=lambda_rhyme).to(device)
    lm.load(modelFileName,device)
    
    stress_dict = load_stress_dict('bg_dict_csv/single_stress.csv')
    
    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generator_char.generateText(lm, char2id, auth, seed, temperature=temperature, stress_predict=stress.predict, stress_dict=stress_dict))
    
if len(sys.argv)>1 and sys.argv[1] == 'generate_transformer':
    if len(sys.argv)>2: auth = sys.argv[2]
    else:
        print('Usage: python run.py generate_transformer author [seed [temperature]]')
    if len(sys.argv)>3: seed = sys.argv[3]
    else: seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>4: temperature = float(sys.argv[4])
    else: temperature = defaultTemperature
 
    tokens2id = pickle.load(open(tokens2idFileName_transformer, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_transformer.TransformerLanguageModelPack(char_emb_size_transformer, hid_size_transformer, auth2id, tokens2id, unkChar, padChar, endChar, n_layers=transformer_layers, n_heads=transformer_heads, dropout=dropout, k_rhyme=k_rhyme, lambda_rhyme=lambda_rhyme).to(device)
    lm.load(modelFileName_transformer,device)
    
    stress_dict = load_stress_dict('bg_dict_csv/single_stress.csv')
    
    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generator_transformer.generateText(lm, tokens2id, auth, seed, temperature=temperature, stress_predict=stress.predict, stress_dict=stress_dict))

if len(sys.argv)>1 and sys.argv[1] == 'generate_token':
    
    print(sys.argv)
    
    if len(sys.argv)>2: auth = sys.argv[2]
    else:
        print('Usage: python run.py generate_token author [debug [seed [temperature]]]')
        
    debug = False
    if len(sys.argv)>3: debug = True
        
    if len(sys.argv)>4: seed = sys.argv[4]
    else: seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>5: temperature = float(sys.argv[5])
    else: temperature = defaultTemperature

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
    print(generate_token.generateText(lm, tokens2id, auth, seed, temperature=temperature, stress_predict=stress.predict, stress_dict=stress_dict, debug=debug))

if len(sys.argv)>1 and sys.argv[1] == 'generate_char_dp':
    if len(sys.argv)>2: auth = sys.argv[2]
    else:
        print('Usage: python run.py generate_char_dp author [seed [temperature]]')
    if len(sys.argv)>3: seed = sys.argv[3]
    else: seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>4: temperature = float(sys.argv[4])
    else: temperature = defaultTemperature

    tokens2id = pickle.load(open(tokens2idFileName_dp, 'rb'))
    auth2id = pickle.load(open(auth2idFileName, 'rb'))
    lm = model_char_dp.CharLSTMLanguageModelPack(
        char_emb_size_dp, hid_size_dp, auth2id, tokens2id,
        unkChar, padChar, endChar, lstm_layers=lstm_layers_dp,
        dropout=dropout_dp, lambda_rhyme=lambda_rhyme_dp
    ).to(device)
    lm.load(modelFileName_dp, device)

    authid = auth2id.get(auth,0)
    if authid==0: print('Авторът не е известен.')
    print(generate_char_dp.generateText(lm, tokens2id, auth, seed, temperature=temperature))
    