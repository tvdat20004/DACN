import base64
def write_data(filename : str, data : bytes):
    with open(filename, 'wb') as file:
        file.write(base64.b64encode(data))

def read_data(filename : str) -> bytes:
    with open(filename, 'rb') as file:
        return base64.b64decode(file.read())

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