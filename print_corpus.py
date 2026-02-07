import argparse
import pickle
import sys

try:
    from parameters import (
        testDataFileName, trainDataFileName,
        testDataFileName_transformer, trainDataFileName_transformer,
        testDataFileName_token, trainDataFileName_token,
    )
except Exception:
    testDataFileName = 'testData'
    trainDataFileName = 'trainData'
    testDataFileName_transformer = 'testDataTransformer'
    trainDataFileName_transformer = 'trainDataTransformer'
    testDataFileName_token = 'testDataToken'
    trainDataFileName_token = 'trainDataToken'


def resolve_paths(variant: str, override_test: str | None, override_train: str | None):
    if override_test or override_train:
        return (
            override_test or testDataFileName,
            override_train or trainDataFileName,
        )
    if variant == 'char':
        return testDataFileName, trainDataFileName
    if variant == 'transformer':
        return testDataFileName_transformer, trainDataFileName_transformer
    if variant == 'token':
        return testDataFileName_token, trainDataFileName_token
    raise ValueError(f"Unknown variant: {variant}")


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def format_sequence(seq):
    try:
        joined = ''.join(seq)
    except Exception:
        joined = ' '.join(map(str, seq))
    return joined


def print_corpus(name: str, corpus, max_items: int | None):
    print(f"=== {name} ({len(corpus)} items) ===")
    count = 0
    for i, (author, seq) in enumerate(corpus):
        if max_items is not None and count >= max_items:
            break
        preview = format_sequence(seq)
        print(f"[{i}] author={author}")
        print(preview)
        print("---")
        count += 1
    if max_items is not None and len(corpus) > max_items:
        print(f"... (truncated, showing first {max_items} of {len(corpus)})")


def main():
    parser = argparse.ArgumentParser(description="Print contents of test/train corpora produced by utils.prepareData")
    parser.add_argument('--variant', choices=['char', 'transformer', 'token'], default='char', help='Corpus variant to load')
    parser.add_argument('--test', dest='test_path', help='Explicit path to test corpus pickle')
    parser.add_argument('--train', dest='train_path', help='Explicit path to train corpus pickle')
    parser.add_argument('--max', dest='max_items', type=int, default=None, help='Limit printed items per corpus')
    args = parser.parse_args()

    test_path, train_path = resolve_paths(args.variant, args.test_path, args.train_path)
    try:
        test_corpus = load_pickle(test_path)
        train_corpus = load_pickle(train_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Hint: run the appropriate prepare step first (e.g., 'python run.py prepare', 'prepare_transformer', or 'prepare_token').")
        sys.exit(1)

    print_corpus("TestCorpus", test_corpus, args.max_items)
    print_corpus("TrainCorpus", train_corpus, args.max_items)


if __name__ == '__main__':
    main()
