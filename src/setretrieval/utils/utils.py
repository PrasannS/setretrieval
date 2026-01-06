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

def check_process_tset_mini(tset):
    if type(tset['pos_chunks'][0]) == dict:
        tset['pos_chunks'] = [r['text'] for r in tset['pos_chunks']]
    return tset

def check_process_tset(tset):
    return tset.map(check_process_tset_mini)

def preds_to_chunks(preds, ds): 
    chunks = []
    for plist in preds: 
        chunks.append([ds[p['index']]['text'] for p in plist])
    return chunks