import base64
import tenseal as ts
try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False

def write_data(filename : str, data : bytes):
    with open(filename, 'wb') as file:
        file.write(base64.b64encode(data))

def read_data(filename : str) -> bytes:
    print("hahaha")
    with open(filename, 'rb') as file:
        data = base64.b64decode(file.read())
        print("hahahahha")
    return data

def _almost_equal_number(v1, v2, m_pow_ten : int = 1) -> bool:
    upper_bound = pow(10, -m_pow_ten)

    return abs(v1 - v2) <= upper_bound


def _almost_equal(vec1, vec2, m_pow_ten : int = 1) -> bool:
    if not isinstance(vec1, list):
        return _almost_equal_number(vec1, vec2, m_pow_ten)

    if len(vec1) != len(vec2):
        return False

    for v1, v2 in zip(vec1, vec2):
        if isinstance(v1, list):
            if not _almost_equal(v1, v2, m_pow_ten):
                return False
        elif not _almost_equal_number(v1, v2, m_pow_ten):
            return False
    return True

context = ts.context_from(read_data("../keys/public.txt"))
def encrypt_matrix(matrix : np.ndarray) -> bytes:
    return ts.ckks_tensor(context, matrix.tolist()).serialize()
