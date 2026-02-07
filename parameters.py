import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'
auth2idFileName = 'auth2id'

device = torch.device("cuda:0")

batchSize = 64
char_emb_size = 128

hid_size = 512
lstm_layers = 3
dropout = 0.6
maxPoemLength = 10000

batchSize_transformer = 128
char_emb_size_transformer = 128
hid_size_transformer = 512
transformer_layers = 4
transformer_heads = 4
maxPoemLength_transformer = 1000
merges_transformer = 200
corpusFileName_transformer = 'corpusPoems'
modelFileName_transformer = 'modelTransformer'
trainDataFileName_transformer = 'trainDataTransformer'
testDataFileName_transformer = 'testDataTransformer'
tokens2idFileName_transformer = 'char2idTransformer'

epochs = 10000
learning_rate = 0.001

defaultTemperature = 0.4

k_rhyme = 3
lambda_rhyme = 0.1

# Token LSTM (BPE tokens) configuration
batchSize_token = 128
char_emb_size_token = 128
hid_size_token = 512
lstm_layers_token = 3
dropout_token = 0.6
maxPoemLength_token = 10000
merges_token = 100
corpusFileName_token = 'corpusPoems'
modelFileName_token = 'modelLSTMToken'
trainDataFileName_token = 'trainDataToken'
testDataFileName_token = 'testDataToken'
tokens2idFileName_token = 'tokens2idToken'
lambda_rhyme_token = 0.1


# Token LSTM (BPE tokens) configuration
batchSize_dp = 128
char_emb_size_dp = 128
hid_size_dp = 512
lstm_layers_dp = 3
dropout_dp = 0.4
maxPoemLength_dp = 10000
corpusFileName_dp = 'corpusPoems'
modelFileName_dp = 'modelLSTMDp'
trainDataFileName_dp = 'trainDataDp'
testDataFileName_dp = 'testDataDp'
tokens2idFileName_dp = 'char2idDp'
lambda_rhyme_dp = 0.0

# Reversed Char LSTM (RTL per-line) configuration
batchSize_reversed = 128
char_emb_size_reversed = 128
hid_size_reversed = 512
lstm_layers_reversed = 3
dropout_reversed = 0.4
maxPoemLength_reversed = 10000
corpusFileName_reversed = 'corpusPoems'
modelFileName_reversed = 'modelLSTMReversed'
trainDataFileName_reversed = 'trainDataDp'      
testDataFileName_reversed = 'testDataDp'        
tokens2idFileName_reversed = 'char2idDp'        
