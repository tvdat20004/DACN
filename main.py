import numpy as np
import tenseal as ts
from client_side.utils import Utils

utils = Utils('./keys')
a = np.random.rand(11, 256)
b = np.random.rand(256, 256)

a_ = utils.encrypt_matrix(a)
b_ = utils.encrypt_matrix(b)

print(a_.shape, b_.shape)

ab_ = a_.dot(b_)
ab = a.dot(b)
print(utils._almost_equal(utils.decrypt_matrix(ab_), ab))
