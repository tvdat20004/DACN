import tenseal as ts
from transformer import utils
try:
    import cupy as np
    is_cupy_available = True
    print('CuPy is available. Using CuPy for all computations.')
except:
    import numpy as np
    is_cupy_available = False
    print('CuPy is not available. Switching to NumPy.')


def encrypt_matrix(matrix : np.ndarray) -> bytes:
    # print(matrix.tolist(), type(matrix.tolist()))
    return ts.ckks_tensor(utils.context, matrix.tolist()).serialize()

# print(type(encrypt_matrix(matrix_data)))