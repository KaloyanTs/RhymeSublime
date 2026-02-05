import random
import re

corpusSplitString = '@\n'
symbolCountThreshold = 100
authorCountThreshold = 20

def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus

def getAlphabetAuthors(corpus):
    symbols={}
    authors={}
    for s in corpus:
        if len(s) > 0:
            n=s.find('\n')
            aut = s[:n]
            if aut in authors: authors[aut] += 1
            else: authors[aut] = 1
            poem = s[n+1:]
            for c in poem:
                if c in symbols: symbols[c] += 1
                else: symbols[c]=1
    return symbols, authors

def _tokenize_text(text: str):
    """
    Simple local tokenizer producing tokens including spaces and newlines.
    - Words: consecutive letters (Latin or Cyrillic)
    - Spaces: ' '
    - Newlines: '\n'
    - Other punctuation: single-character tokens
    """
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\n':
            tokens.append('\n')
            i += 1
        elif ch == ' ':
            tokens.append(' ')
            i += 1
        else:
            # word token: letters only
            if re.match(r"[A-Za-z\u0400-\u04FF\u0500-\u052F]", ch):
                j = i + 1
                while j < len(text) and re.match(r"[A-Za-z\u0400-\u04FF\u0500-\u052F]", text[j]):
                    j += 1
                tokens.append(text[i:j])
                i = j
            else:
                # punctuation or other symbol
                tokens.append(ch)
                i += 1
    return tokens

def _split_units(text: str):
    """Split into units: words (letters) and single-character separators (space, newline, punctuation)."""
    units = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\n' or ch == ' ':
            units.append(ch)
            i += 1
        elif re.match(r"[A-Za-z\u0400-\u04FF\u0500-\u052F]", ch):
            j = i + 1
            while j < len(text) and re.match(r"[A-Za-z\u0400-\u04FF\u0500-\u052F]", text[j]):
                j += 1
            units.append(text[i:j])
            i = j
        else:
            units.append(ch)
            i += 1
    return units

def _bpe_train(words, merges: int):
    """Train simple BPE merges on a list of words (each word is a string)."""
    # Represent words as lists of characters
    vocab = [list(w) for w in words if w]

    def count_pairs(vocab_seq):
        pairs = {}
        for symbols in vocab_seq:
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs[pair] + 1 if pair in pairs else 1
        return pairs

    merges_list = []
    for _ in range(max(0, merges)):
        print("BPE training merge step:", _ + 1, "of", merges)
        pairs = count_pairs(vocab)
        if not pairs:
            break
        best = max(pairs.items(), key=lambda x: x[1])[0]
        print("Best pair:", best, "Count:", pairs[best])
        merges_list.append(best)

        # Merge occurrences of best pair
        a, b = best
        new_vocab = []
        for symbols in vocab:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_vocab.append(new_symbols)
        vocab = new_vocab

    return merges_list

def _bpe_encode(word: str, merges_list):
    """Encode a single word using trained merges list (applied left-to-right greedily)."""
    symbols = list(word)
    if not merges_list:
        return symbols
    # Apply merges in order
    for a, b in merges_list:
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(a + b)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols

def _greedy_encode(word: str, vocab_tokens):
    """Greedy longest-prefix encoding of a word using final token set.
    Assumes vocab_tokens is a list of tokens sorted by length descending.
    Falls back to single-character token when no longer token matches.
    """
    if not word:
        return []
    i = 0
    out = []
    L = len(word)
    while i < L:
        matched = False
        for tok in vocab_tokens:
            tl = len(tok)
            if tl == 0 or i + tl > L:
                continue
            if word[i:i+tl] == tok:
                out.append(tok)
                i += tl
                matched = True
                break
        if not matched:
            out.append(word[i])
            i += 1
    return out


def prepareData(corpusFileName, startChar, endChar, unkChar, padChar, maxPoemLength, tokenization = 'char', bpe_merges: int = 500):
    print('[Utils] Loading and preparing data...')
    print('[Utils] Mode:', tokenization, '| bpe_merges:', bpe_merges if tokenization == 'bpe' else 'n/a')
    file = open(corpusFileName,'r', encoding='utf-8')
    print('[Utils] Reading corpus file:', corpusFileName)
    poems = file.read().split(corpusSplitString)
    print('[Utils] Extracting alphabet and authors...')
    symbols, authors = getAlphabetAuthors(poems)
    print(f"[Utils] Totals: poems={len(poems):,}, symbols={len(symbols):,}, authors={len(authors):,}")
    
    assert startChar not in symbols and endChar not in symbols and unkChar not in symbols and padChar not in symbols

    if tokenization == 'char':
        charset = [startChar,endChar,unkChar,padChar] + [c for c in sorted(symbols) if symbols[c] > symbolCountThreshold]
        char2id = { c:i for i,c in enumerate(charset)}
        print('[Utils] Char vocab size:', len(charset))
    elif tokenization == 'token':
        # Build token vocabulary from poems with local tokenizer
        token_counts = {}
        for i,s in enumerate(poems):
            print(f"Processing poem {i+1} for tokenization...", end='\r', flush=True)
            if len(s) > 0:
                n = s.find('\n')
                poem = s[n+1:]
                for tok in _tokenize_text(poem):
                    token_counts[tok] = token_counts.get(tok, 0) + 1
        # Ensure special tokens are present
        for sp in (startChar, endChar, unkChar, padChar):
            token_counts[sp] = token_counts.get(sp, 0)
        # Vocabulary with threshold
        tokenset = [startChar,endChar,unkChar,padChar] + [t for t in sorted(token_counts) if token_counts[t] > symbolCountThreshold]
        char2id = { t:i for i,t in enumerate(tokenset)}
        print('[Utils] Token vocab size:', len(tokenset))
    elif tokenization == 'bpe':
        # Train BPE merges over words from corpus
        print("[Utils] Training BPE merges...")
        all_words = []
        for i,s in enumerate(poems):
            print(f"Processing poem {i+1} for BPE training...", end='\r', flush=True)
            if len(s) > 0:
                n = s.find('\n')
                poem = s[n+1:]
                units = _split_units(poem)
                for u in units:
                    if u and u not in (' ', '\n') and re.match(r"^[A-Za-z\u0400-\u04FF\u0500-\u052F]+$", u):
                        all_words.append(u)
        merges_list = _bpe_train(all_words, merges=bpe_merges)
        print('[Utils] BPE merges learned:', len(merges_list))

        token_counts = {}
        # Build tokens for entire corpus using trained merges
        for i,s in enumerate(poems):
            print(f"Processing poem {i+1} for BPE tokenization...", end='\r', flush=True)
            if len(s) > 0:
                n = s.find('\n')
                poem = s[n+1:]
                units = _split_units(poem)
                seq_tokens = []
                for u in units:
                    if u == ' ' or u == '\n':
                        seq_tokens.append(u)
                    elif re.match(r"^[A-Za-z\u0400-\u04FF\u0500-\u052F]+$", u):
                        seq_tokens.extend(_bpe_encode(u, merges_list))
                    else:
                        seq_tokens.append(u)
                for t in seq_tokens:
                    token_counts[t] = token_counts.get(t, 0) + 1
        # Ensure special tokens and ALL single characters are present in vocab, regardless of frequency
        for sp in (startChar, endChar, unkChar, padChar, ' ', '\n'):
            token_counts[sp] = token_counts.get(sp, 0)
        for ch in symbols.keys():
            token_counts[ch] = token_counts.get(ch, 0)

        tokenset = [startChar, endChar, unkChar, padChar] + sorted(token_counts.keys())
        # Deduplicate while preserving the specials order
        seen = set()
        dedup_tokens = []
        for t in tokenset:
            if t not in seen:
                seen.add(t)
                dedup_tokens.append(t)
        char2id = { t:i for i,t in enumerate(dedup_tokens)}
        print('[Utils] BPE token vocab size (with chars):', len(dedup_tokens))
        # Prepare greedy vocab (only letter tokens) sorted by length desc for encoding
        letter_re = re.compile(r"^[A-Za-z\u0400-\u04FF\u0500-\u052F]+$")
        greedy_vocab = [t for t in dedup_tokens if letter_re.match(t)]
        greedy_vocab.sort(key=len, reverse=True)
        # cache merges list in mapping for later use (optional external usage not required here)
        # Note: merges_list isn't persisted; encoding is done during corpus build below
    else:
        raise ValueError("Unsupported tokenization: " + str(tokenization))
    authset = [a for a in sorted(authors) if authors[a] > authorCountThreshold]
    auth2id = { a:i for i,a in enumerate(authset)}
    print('[Utils] Author vocab size:', len(authset))
    
    corpus = []
    for i,s in enumerate(poems):
        if len(s) > 0:
            n=s.find('\n')
            aut = s[:n]
            poem = s[n+1:]
            if tokenization == 'char':
                seq = [startChar] + [ poem[i] for i in range(min(len(poem),maxPoemLength)) ] + [endChar]
            elif tokenization == 'token':
                toks = _tokenize_text(poem)
                seq = [startChar] + toks[:maxPoemLength] + [endChar]
            elif tokenization == 'bpe':
                units = _split_units(poem)
                seq_tokens = []
                # Use trained merges_list from above to encode words
                for u in units:
                    if u == ' ' or u == '\n':
                        seq_tokens.append(u)
                    elif re.match(r"^[A-Za-z\u0400-\u04FF\u0500-\u052F]+$", u):
                        # Greedy longest-match over final token set
                        seq_tokens.extend(_greedy_encode(u, greedy_vocab))
                    else:
                        seq_tokens.append(u)
                seq = [startChar] + seq_tokens[:maxPoemLength] + [endChar]
            else:
                seq = [startChar] + [ poem[i] for i in range(min(len(poem),maxPoemLength)) ] + [endChar]
            corpus.append( (aut, seq) )
            print('[Utils] Corpus sequences built:', len(corpus))

    testCorpus, trainCorpus  = splitSentCorpus(corpus, testFraction = 0.01)
    print('[Utils] Split:', 'train='+str(len(trainCorpus)), 'test='+str(len(testCorpus)))
    print('[Utils] Corpus loading completed.')
    return testCorpus, trainCorpus, char2id, auth2id
