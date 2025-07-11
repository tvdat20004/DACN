try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False

from transformer.layers.base.embedding import Embedding
from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.encoder_layer import EncoderLayer
from transformer.layers.combined.positional_encoding import PositionalEncoding
from typing import List
import tenseal as ts


class Encoder:
    def __init__(self, heads_num : int, layers_num : int, d_model : int, d_ff : int, dropout : float, max_length : int = 5000, data_type : type = np.float32):

        # self.token_embedding    = Embedding(src_vocab_size, d_model, data_type)
        # self.position_embedding = PositionalEncoding(max_length, d_model, dropout, data_type)

        self.layers : List[EncoderLayer] = []
        for _ in range(layers_num):
            self.layers.append(EncoderLayer(d_model, heads_num, d_ff, dropout, data_type))

        # self.dropout = Dropout(dropout, data_type)
        # self.scale = np.sqrt(d_model).astype(data_type)

    def forward(self, src : ts.CKKSTensor, src_mask : np.ndarray):

        # src = self.token_embedding.forward(src) * self.scale
        # src = self.position_embedding.forward(src)
        # src = self.dropout.forward(src, training)
        # //////
        for layer in self.layers:
            src = layer.forward(src, src_mask)

        return src

    def backward(self, error):

        for layer in reversed(self.layers):
            error = layer.backward(error)

        error = self.dropout.backward(error)
        error = self.position_embedding.backward(error) * self.scale
        error = self.token_embedding.backward(error)


    def set_optimizer(self, optimizer):
        self.token_embedding.set_optimizer(optimizer)

        for layer in self.layers:
            layer.set_optimizer(optimizer)

    def update_weights(self):
        layer_num = 1
        layer_num = self.token_embedding.update_weights(layer_num)

        for layer in self.layers:
            layer_num = layer.update_weights(layer_num)
