import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
import pickle
# import numpy as np
try:
    import cupy as cp
    is_cupy_available = True
    print('CuPy is available. Using CuPy for all computations.')
except:
    is_cupy_available = False
    print('CuPy is not available. Switching to NumPy.')
import tenseal as ts

# Tạo context CKKS
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 20, 40])

context.generate_galois_keys()
    
context.global_scale = 2**40

encoder = pickle.load(open("saved models/seq2seq_model/10/encoder.pkl", "rb"))
decoder = pickle.load(open("saved models/seq2seq_model/10/decoder.pkl", "rb"))
# print(encoder.token_embedding.w.shape)
enc = ts.ckks_tensor(context, encoder.token_embedding.w.get())
