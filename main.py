import numpy as np
import tenseal as ts
from client_side.utils import Utils
from key_gen import *
import sys
import time
file = open('log.txt', 'w')

n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else '1'
cxt = parallel_context(n_threads)
secret = cxt.serialize(save_secret_key=True)
write_data("./keys/secret.txt", secret)
cxt.make_context_public()
public = cxt.serialize()
write_data("./keys/public.txt", public)
file.write("Keys generated and saved to ./keys/secret.txt and ./keys/public.txt\n")
print("Keys generated and saved to ./keys/secret.txt and ./keys/public.txt\n")

utils = Utils("./keys")
a = np.random.rand(11, 256)
b = np.random.rand(256, 256)
now = time.time()
a_ = utils.encrypt_matrix(a)
b_ = utils.encrypt_matrix(b)

file.write(f"Complete encryption\n{time.time() - now}\n")
now = time.time()

ab_ = a_.dot(b_)
ab = a.dot(b)
res = utils._almost_equal(utils.decrypt_matrix(ab_), ab)
file.write(f"Result: {res}\n{time.time() - now}\n")
