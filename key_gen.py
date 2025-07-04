import tenseal as ts
import base64
def context() -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree = 8192,
        coeff_mod_bit_sizes = [31, 26,26 ,26 ,26 ,26 ,26 ,31]
    )
    context.generate_galois_keys()
    context.global_scale = pow(2, 26)
    return context
def parallel_context(n_threads : int) -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[31, 26,26 ,26 ,26 ,26 ,26 ,31], n_threads=n_threads
    )
    context.global_scale = pow(2, 26)
    context.generate_galois_keys()
    return context

def write_data(filename : str, data : bytes):
    with open(filename, 'wb') as file:
        file.write(base64.b64encode(data))

if __name__ == "__main__":
    cxt = parallel_context(12)
    secret = cxt.serialize(save_secret_key=True)
    write_data("./keys/secret.txt", secret)
    cxt.make_context_public()
    public = cxt.serialize()
    write_data("./keys/public.txt", public)
    print("Keys generated and saved to ./keys/secret.txt and ./keys/public.txt")
