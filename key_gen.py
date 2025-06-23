import tenseal as ts
import utils

def context() -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree = 8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = pow(2, 40)
    return context
def parallel_context(n_threads : int) -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=n_threads
    )
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context
if __name__ == "__main__":
    cxt = context()
    secret = cxt.serialize(save_secret_key=True)
    utils.write_data("./keys/secret.txt", secret)
    cxt.make_context_public()
    public = cxt.serialize()
    utils.write_data("./keys/public.txt", public)
    print("Keys generated and saved to ./keys/secret.txt and ./keys/public.txt")
