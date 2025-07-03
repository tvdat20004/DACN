import base64
import tenseal as ts
try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False
from typing import List, Union
class Utils:
    def __init__(self, keys_folder_path : str):
        self.public_context = ts.context_from(Utils.read_data(f"{keys_folder_path}/public.txt"))
        self.secret_context = ts.context_from(Utils.read_data(f"{keys_folder_path}/secret.txt"))
        self.secret_key = self.secret_context.secret_key()
    @staticmethod
    def ndim(X : ts.CKKSTensor) -> int:
        return len(X.shape)
    @staticmethod
    def add(tensor1: Union[ts.CKKSTensor, List[ts.CKKSTensor]], tensor2: ts.CKKSTensor) -> ts.CKKSTensor:
        if isinstance(tensor1, list):
            return [tensor1[i] + tensor2 for i in range(len(tensor1))]
        elif isinstance(tensor1, ts.CKKSTensor):
            if Utils.ndim(tensor1) == 3 and Utils.ndim(tensor2) == 2:
                return [tensor1[i].reshape(tensor1.shape[1:]) + tensor2 for i in range(tensor1.shape[0])]
        else:
            return tensor1 + tensor2
    @staticmethod
    def _dot_2d(tensor1: ts.CKKSTensor, tensor2: ts.CKKSTensor) -> ts.CKKSTensor:
        return tensor1.dot(tensor2)

    @staticmethod
    def _dot_3d_2d(tensor_3d: ts.CKKSTensor, tensor_2d: ts.CKKSTensor) -> List[ts.CKKSTensor]:

        if tensor_3d.shape[2] != tensor_2d.shape[0]:
            raise ValueError(
                f"Can't perform a dot product between 2 matrix: {tensor_3d.shape}  {tensor_2d.shape}"
            )
        result_slices = []
        num_slices = tensor_3d.shape[0]
        for i in range(num_slices):
            slice_2d = tensor_3d[i].reshape(list(tensor_3d.shape[1:]))
            # print(f"Slice {i} shape: {slice_2d.shape}, Tensor 2D shape: {tensor_2d.shape}")
            result_slice = Utils._dot_2d(slice_2d, tensor_2d)
            result_slices.append(result_slice)
        return result_slices
    @staticmethod
    def dot(tensor1: ts.CKKSTensor, tensor2: ts.CKKSTensor) -> Union[ts.CKKSTensor, List[ts.CKKSTensor]]:
        """
        Simulate the dot product operation between two CKKSTensors,
        similar to numpy.dot().
        If tensor1 is 3D and tensor2 is 2D, it performs a batch dot product, return a list of CKKSTensors.
        If tensor1 is 2D and tensor2 is 2D, it performs a single dot product and returns a single CKKSTensor.
        """
        if Utils.ndim(tensor1) == 3 and Utils.ndim(tensor2) == 2:
            return Utils._dot_3d_2d(tensor1, tensor2)
        else:
            return Utils._dot_2d(tensor1, tensor2)

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
    def encrypt_matrix(self, matrix : np.ndarray) -> ts.CKKSTensor:
        return ts.ckks_tensor(self.public_context, matrix.tolist())
    def decrypt_matrix(self, encrypted_matrix : ts.CKKSTensor) -> np.ndarray:
        return np.array(encrypted_matrix.decrypt(self.secret_key).tolist(), dtype=np.float32)
