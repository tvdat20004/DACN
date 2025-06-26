try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False
from typing import Optional, List
import tenseal as ts
from transformer import utils
class Dense():
    """
    Add Dense layer
    ---------------
        Args:
            `units_num` (int): number of neurons in the layer
            `use_bias` (bool):  `True` if used. `False` if not used
        Returns:
            output: data with shape (batch_size, units_num)
    """

    def __init__(self, encrypted_weight : List[ts.CKKSTensor], units_num : int, inputs_num : Optional[int] = None, use_bias : bool = True, data_type : type = np.float32) -> None:

        # self.units_num   = units_num
        # self.inputs_num = inputs_num
        # self.use_bias    = use_bias

        self.w = None
        self.b = None

        # self.optimizer = None
        # self.data_type = data_type
        self.set_encrypted_weights(encrypted_weight)
        # self.build()

    # def set_optimizer(self, optimizer):
    #     self.optimizer = optimizer

    def set_encrypted_weights(self, weights : List[ts.CKKSTensor]) -> None:
        self.w, self.b, self.grad_b, self.grad_w= weights

    # def build(self) -> None:

    #     stdv = 1. / np.sqrt(self.inputs_num)# * 0.5 #input size

    #     self.w = np.random.uniform(-stdv, stdv, (self.inputs_num, self.units_num)).astype(self.data_type)

    #     self.b = np.zeros(self.units_num).astype(self.data_type)


    #     self.v, self.m         = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type) # optimizers params
    #     self.v_hat, self.m_hat = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type) # optimizers params

    #     self.vb, self.mb         = np.zeros_like(self.b).astype(self.data_type), np.zeros_like(self.b).astype(self.data_type) # optimizers params
    #     self.vb_hat, self.mb_hat = np.zeros_like(self.b).astype(self.data_type), np.zeros_like(self.b).astype(self.data_type) # optimizers params

    #     self.output_shape = (1, self.units_num)

    def forward(self, X : ts.CKKSTensor) -> ts.CKKSTensor:
        self.input_data = X

        self.batch_size = self.input_data.shape[0]
        self.output_data = self.input_data * self.w + self.b
        return self.output_data

    def backward(self, error):
        self.grad_w = np.sum(np.matmul(self.input_data.transpose(0, 2, 1), error), axis = 0)
        self.grad_b = np.sum(error, axis = (0, 1))

        output_error = np.dot(error, self.w.T)

        return output_error

    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)
        if self.use_bias == True:
            self.b, self.vb, self.mb, self.vb_hat, self.mb_hat  = self.optimizer.update(self.grad_b, self.b, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)

        return layer_num + 1

    # def get_grads(self):
    #     return self.grad_w, self.grad_b

    # def set_grads(self, grads):
    #     self.grad_w, self.grad_b = grads
