import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
import pickle
from transformer.prepare_data import DataPreparator

import numpy as np
try:
    import cupy as cp
    is_cupy_available = True
    print('CuPy is available. Using CuPy for all computations.')
except:
    is_cupy_available = False
    print('CuPy is not available. Switching to NumPy.')
import tenseal as ts

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
