import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


import numpy as np
try:
    import cupy as cp
    is_cupy_available = True
    print('CuPy is available. Using CuPy for all computations.')
except:
    is_cupy_available = False
    print('CuPy is not available. Switching to NumPy.')

import pickle as pkl
from tqdm import tqdm
import tenseal as ts
from transformer.modules import Encoder
from transformer.modules import Decoder
from transformer.optimizers import Adam, Nadam, Momentum, RMSProp, SGD, Noam
from transformer.losses import CrossEntropy
from transformer.prepare_data import DataPreparator
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from transformer.utils import Utils

unsplited_mapping = Dict[str, str]
splited_mapping = Dict[str, List[str]]


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
tokens  = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
indexes = (PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX)

# data_preparator = DataPreparator(tokens, indexes)

# train_data, test_data, val_data = data_preparator.prepare_data(
#                     path = dataset_path,
#                     batch_size = BATCH_SIZE,
#                     min_freq = 2)

# source, target = train_data

# train_data_vocabs = data_preparator.get_vocabs()



class Seq2Seq():

    def __init__(self, encoder : Encoder, decoder : Decoder, pad_idx : int) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

        self.optimizer = Adam()
        self.loss_function = CrossEntropy()

    def set_optimizer(self):
        encoder.set_optimizer(self.optimizer)
        decoder.set_optimizer(self.optimizer)

    def compile(self, optimizer : Noam, loss_function : CrossEntropy) -> None:
        self.optimizer = optimizer
        self.loss_function = loss_function


    def load(self, path : str) -> None:
        pickle_encoder = open(f'{path}/encoder.pkl', 'rb')
        pickle_decoder = open(f'{path}/decoder.pkl', 'rb')
        # print(type(self.encoder.dropout.scale))

        self.encoder = pkl.load(pickle_encoder)
        self.decoder = pkl.load(pickle_decoder)
        pickle_encoder.close()
        pickle_decoder.close()

        print(f'Loaded from "{path}"')

    def save(self, path : str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        pickle_encoder = open(f'{path}/encoder.pkl', 'wb')
        pickle_decoder = open(f'{path}/decoder.pkl', 'wb')

        pkl.dump(self.encoder, pickle_encoder)
        pkl.dump(self.decoder, pickle_decoder)

        pickle_encoder.close()
        pickle_decoder.close()

        print(f'Saved to "{path}"')

    def get_pad_mask(self, x : np.ndarray) -> np.ndarray:
        #x: (batch_size, seq_len)
        return (x != self.pad_idx).astype(int)[:, np.newaxis, :]

    def get_sub_mask(self, x):
        #x: (batch_size, seq_len)
        seq_len = x.shape[1]
        subsequent_mask = np.triu(np.ones((seq_len, seq_len)), k = 1).astype(int)
        subsequent_mask = np.logical_not(subsequent_mask)
        return subsequent_mask

    def forward(self, src, trg, training):
        src, trg = src.astype(DATA_TYPE), trg.astype(DATA_TYPE)
        #src: (batch_size, source_seq_len)
        #tgt: (batch_size, target_seq_len)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)
        src_mask = self.get_pad_mask(src)

        trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)

        enc_src = self.encoder.forward(src, src_mask, training)

        out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask, training)
        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, heads_num, target_seq_len, source_seq_len)
        return out, attention

    def backward(self, error):
        error = self.decoder.backward(error)
        error = self.encoder.backward(self.decoder.encoder_error)

    def update_weights(self):
        self.encoder.update_weights()
        self.decoder.update_weights()

    def _train(self, source, target, epoch, epochs):
        loss_history = []

        tqdm_range = tqdm(enumerate(zip(source, target)), total = len(source))
        for batch_num, (source_batch, target_batch) in tqdm_range:

            output, attention = self.forward(source_batch, target_batch[:,:-1], training = True)

            _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

            loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].astype(np.int32).flatten()).mean())#[:, np.newaxis]
            error = self.loss_function.derivative(_output, target_batch[:, 1:].astype(np.int32).flatten())#[:, np.newaxis]


            self.backward(error.reshape(output.shape))
            self.update_weights()

            tqdm_range.set_description(
                    f"training | loss: {loss_history[-1]:.7f} | perplexity: {np.exp(loss_history[-1]):.7f} | epoch {epoch + 1}/{epochs}" #loss: {loss:.4f}
                )

            if batch_num == (len(source) - 1):
                if is_cupy_available:
                    epoch_loss = cp.mean(cp.array(loss_history))
                else:
                    epoch_loss = np.mean(loss_history)

                tqdm_range.set_description(
                        f"training | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f} | epoch {epoch + 1}/{epochs}"
                )

        return epoch_loss.get() if is_cupy_available else epoch_loss

    def _evaluate(self, source, target):
        loss_history = []

        tqdm_range = tqdm(enumerate(zip(source, target)), total = len(source))
        for batch_num, (source_batch, target_batch) in tqdm_range:

            output, attention = self.forward(source_batch, target_batch[:,:-1], training = False)

            _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

            loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].astype(np.int32).flatten()).mean())

            tqdm_range.set_description(
                    f"testing  | loss: {loss_history[-1]:.7f} | perplexity: {np.exp(loss_history[-1]):.7f}"
                )

            if batch_num == (len(source) - 1):
                if is_cupy_available:
                    epoch_loss = cp.mean(cp.array(loss_history))
                else:
                    epoch_loss = np.mean(loss_history)

                tqdm_range.set_description(
                        f"testing  | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f}"
                )

        return epoch_loss.get() if is_cupy_available else epoch_loss

    # Fit the model on the training data and evaluate on the validation data
    def fit(self, train_data, val_data, epochs, save_every_epochs, save_path = None, validation_check = False):
        self.set_optimizer()

        best_val_loss = float('inf')

        train_loss_history = []
        val_loss_history = []

        train_source, train_target = train_data
        val_source, val_target = val_data

        for epoch in range(epochs):

            train_loss_history.append(self._train(train_source, train_target, epoch, epochs))
            val_loss_history.append(self._evaluate(val_source, val_target))


            if (save_path is not None) and ((epoch + 1) % save_every_epochs == 0):
                if validation_check == False:
                    self.save(save_path + f'/{epoch + 1}')
                else:
                    if val_loss_history[-1] < best_val_loss:
                        best_val_loss = val_loss_history[-1]

                        self.save(save_path + f'/{epoch + 1}')
                    else:
                        print(f'Current validation loss is higher than previous; Not saved')

        return train_loss_history, val_loss_history


# def(predict(self, ckks_tensor, max_length)))

    def predict(self, enc_tensor : ts.CKKSTensor, src_mask : np.array, max_length : int = 50):

        # src_inds = [vocabs[0][word] if word in vocabs[0] else UNK_INDEX for word in sentence]
        # src_inds = [SOS_INDEX] + src_inds + [EOS_INDEX]

        # src = np.asarray(src_inds).reshape(1, -1)
        # src_mask =  self.get_pad_mask(src)
        # print(src_mask, type(src_mask))

        enc_src = self.encoder.forward(enc_tensor, src_mask)

        trg_inds = [SOS_INDEX]

        for _ in range(max_length):
            trg = np.asarray(trg_inds).reshape(1, -1)
            trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)

            out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask, training = False)

            trg_indx = out.argmax(axis=-1)[:, -1].item()
            trg_inds.append(trg_indx)

            if trg_indx == EOS_INDEX or len(trg_inds) >= max_length:
                break

        enc_trg_inds = trg_inds
        return enc_trg_inds
        # reversed_vocab = dict((v,k) for k,v in vocabs[1].items())
        # decoded_sentence = [reversed_vocab[indx] if indx in reversed_vocab else UNK_TOKEN for indx in trg_inds]

        # return decoded_sentence[1:], attention[0]




# INPUT_DIM = len(train_data_vocabs[0])
# OUTPUT_DIM = len(train_data_vocabs[1])
HID_DIM = 256  #512 in original paper
ENC_LAYERS = 3 #6 in original paper
DEC_LAYERS = 3 #6 in original paper
ENC_HEADS = 8
DEC_HEADS = 8
FF_SIZE = 512  #2048 in original paper
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

MAX_LEN = 5000


encoder = Encoder(ENC_HEADS, ENC_LAYERS, HID_DIM, FF_SIZE, ENC_DROPOUT, MAX_LEN, DATA_TYPE)
decoder = Decoder(DEC_HEADS, DEC_LAYERS, HID_DIM, FF_SIZE, DEC_DROPOUT, MAX_LEN, DATA_TYPE)
decoder = None


model = Seq2Seq(encoder, decoder, PAD_INDEX)

try:
    model.load("../saved models/seq2seq_model/10")
except:
    print("Can't load saved model state")

model.compile(
                optimizer = Noam(
                                Adam(alpha = 1e-4, beta = 0.9, beta2 = 0.98, epsilon = 1e-9), #NOTE: alpha doesn`t matter for Noam scheduler
                                model_dim = HID_DIM,
                                scale_factor = 2,
                                warmup_steps = 4000
                            )
                , loss_function = CrossEntropy(ignore_index=PAD_INDEX)
            )
train_loss_history, val_loss_history = None, None
# train_loss_history, val_loss_history = model.fit(train_data, val_data, epochs = 30, save_every_epochs = 5, save_path = "saved models/seq2seq_model", validation_check = True)# "saved models/seq2seq_model"


# def plot_loss_history(train_loss_history, val_loss_history):
#     plt.plot(train_loss_history)
#     plt.plot(val_loss_history)
#     plt.title('Loss history')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.show()

# if train_loss_history is not None and val_loss_history is not None:
#     plot_loss_history(train_loss_history, val_loss_history)
utils = Utils("../keys")
data_path = "../enc_data"

enc = Utils.read_data(f'{data_path}/enc_data')
mask = Utils.read_data(f'{data_path}/enc_mask')

enc_tensor = ts.ckks_tensor_from(utils.public_context, enc)
mask = np.frombuffer(mask, dtype=int)
response = model.predict(enc_tensor, mask)



# _, _, val_data = data_preparator.import_multi30k_dataset(path = dataset_path)
# val_data = data_preparator.clear_dataset(val_data)[0]
# sentences_num = 10

# random_indices = np.random.randint(0, len(val_data), sentences_num)
# sentences_selection : List[splited_mapping] = [val_data[i] for i in random_indices]

#Translate sentences from validation set
# for i, example in enumerate(sentences_selection):
#     print(f"\nExample №{i + 1}")
#     print(f"Input sentence: { ' '.join(example['en'])}")
#     print(f"Decoded sentence: {' '.join(model.predict(example['en'], train_data_vocabs)[0])}")
#     print(f"Target sentence: {' '.join(example['de'])}")




# def plot_attention(sentence, translation, attention, heads_num = 8, rows_num = 2, cols_num = 4):

#     assert rows_num * cols_num == heads_num

#     sentence = [SOS_TOKEN] + [word.lower() for word in sentence] + [EOS_TOKEN]

#     fig = plt.figure(figsize = (15, 25))

#     for h in range(heads_num):

#         ax = fig.add_subplot(rows_num, cols_num, h + 1)
#         ax.set_xlabel(f'Head {h + 1}')

#         if is_cupy_available:
#             ax.matshow(cp.asnumpy(attention[h]), cmap = 'inferno')
#         else:
#             ax.matshow(attention[h], cmap = 'inferno')

#         ax.tick_params(labelsize = 7)

#         ax.set_xticks(range(len(sentence)))
#         ax.set_yticks(range(len(translation)))

#         ax.set_xticklabels(sentence, rotation=90)
#         ax.set_yticklabels(translation)


#     plt.show()

#Plot Attention
# sentence = sentences_selection[0]['en']#['a', 'trendy', 'girl', 'talking', 'on', 'her', 'cellphone', 'while', 'gliding', 'slowly', 'down', 'the', 'street']
# print(f"\nInput sentence: {sentence}")
# decoded_sentence, attention =  model.predict(sentence, train_data_vocabs)
# print(f"Decoded sentence: {decoded_sentence}")

# plot_attention(sentence, decoded_sentence, attention)