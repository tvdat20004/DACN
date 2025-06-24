try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False
from dropout import Dropout
from embedding import Embedding
from positional_encoding import PositionalEncoding
from prepare_data import DataPreparator
from encrypt import encrypt_matrix
import utils

# Get data
DATA_TYPE = np.float32
BATCH_SIZE = 32

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3
dataset_path = '../dataset/'
tokens  = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
indexes = (PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX)

data_preparator = DataPreparator(tokens, indexes)
train_data, test_data, val_data = data_preparator.prepare_data(
                    path = dataset_path,
                    batch_size = BATCH_SIZE,
                    min_freq = 2)

source, target = train_data
train_data_vocabs = data_preparator.get_vocabs()

_, _, val_data = data_preparator.import_multi30k_dataset(path = dataset_path)
val_data = data_preparator.clear_dataset(val_data)[0]

# Convert to vector
INPUT_DIM = len(train_data_vocabs[0])
OUTPUT_DIM = len(train_data_vocabs[1])
HID_DIM = 256  #512 in original paper
ENC_LAYERS = 3 #6 in original paper
DEC_LAYERS = 3 #6 in original paper
ENC_HEADS = 8
DEC_HEADS = 8
FF_SIZE = 512  #2048 in original paper
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
def get_pad_mask(x : np.ndarray) -> np.ndarray:
    #x: (batch_size, seq_len)
    return (x != PAD_INDEX).astype(int)[:, np.newaxis, :]

token_embedding    = Embedding(INPUT_DIM, HID_DIM, np.float32)
position_embedding = PositionalEncoding(5000, HID_DIM, ENC_DROPOUT, np.float32)
dropout = Dropout(ENC_DROPOUT, np.float32)
scale = np.sqrt(HID_DIM).astype(np.float32)

idx = 0 # Choose random index
vocabs = train_data_vocabs
sentence = val_data[idx]['en']
src_inds = [vocabs[0][word] if word in vocabs[0] else UNK_INDEX for word in sentence]
src_inds = [SOS_INDEX] + src_inds + [EOS_INDEX]
src = np.asarray(src_inds).reshape(1, -1)
src = token_embedding.forward(src) * scale
src = position_embedding.forward(src)
src = dropout.forward(src, training = False)

# Encrypt ?

enc_src = encrypt_matrix(src)
utils.write_data("../enc_data/enc_data", enc_src)