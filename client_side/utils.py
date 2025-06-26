import base64
import tenseal as ts
try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False
class Utils:
    def __init__(self, keys_folder_path : str):
        self.public_context = ts.context_from(Utils.read_data(f"{keys_folder_path}/public.txt"))
        self.secret_context = ts.context_from(Utils.read_data(f"{keys_folder_path}/secret.txt"))
    @staticmethod
    def write_data(filename : str, data : bytes):
        with open(filename, 'wb') as file:
            file.write(base64.b64encode(data))

    @staticmethod
    def read_data(filename : str) -> bytes:
        with open(filename, 'rb') as file:
            return base64.b64decode(file.read())

    @staticmethod
    def _almost_equal_number(v1, v2, m_pow_ten : int = 1) -> bool:
        upper_bound = pow(10, -m_pow_ten)

        return abs(v1 - v2) <= upper_bound

    @staticmethod
    def _almost_equal(vec1, vec2, m_pow_ten : int = 1) -> bool:
        if not isinstance(vec1, list):
            return Utils._almost_equal_number(vec1, vec2, m_pow_ten)

        if len(vec1) != len(vec2):
            return False

        for v1, v2 in zip(vec1, vec2):
            if isinstance(v1, list):
                if not Utils._almost_equal(v1, v2, m_pow_ten):
                    return False
            elif not Utils._almost_equal_number(v1, v2, m_pow_ten):
                return False
        return True
    def encrypt_matrix(self, matrix : np.ndarray) -> bytes:
        return ts.ckks_tensor(self.public_context, matrix.tolist())
