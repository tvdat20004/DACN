from transformer_raw.modules import encoder, decoder
import cupy as np
import tenseal as ts
from client_side.utils import Utils
from key_gen import *
import sys
import time
import pickle as pkl
def load(path : str) -> None:
	pickle_encoder = open(f'{path}/encoder.pkl', 'rb')
	pickle_decoder = open(f'{path}/decoder.pkl', 'rb')
	# print(type(self.encoder.dropout.scale))

	encoder = pkl.load(pickle_encoder)
	decoder = pkl.load(pickle_decoder)
	pickle_encoder.close()
	pickle_decoder.close()
	return encoder, decoder
file = open('log.txt', 'w')

path = "./saved_models/seq2seq_model/10"
encoder, decoder = load(path)
file.write(f'Loaded encoder and decoder from {path}\n')

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
a = np.random.rand(11, 12)
b = np.random.rand(12, 12)
now = time.time()
a_ = utils.encrypt_matrix(a)
b_ = utils.encrypt_matrix(b)

file.write(f"Complete encryption\n{time.time() - now}\n")
now = time.time()

ab_ = a_.dot(b_)
ab = a.dot(b)
res = utils._almost_equal(utils.decrypt_matrix(ab_), ab)
file.write(f"Result: {res}\n{time.time() - now}\n")
