import pickle
import hashlib

def get_deterministic_hash(input_string):
    # Ensure the input is encoded to bytes
    encoded_str = input_string.encode('utf-8')
    # Use SHA-256 for a robust, deterministic hash
    return hashlib.sha256(encoded_str).hexdigest()


def pickdump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)