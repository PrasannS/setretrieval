import pickle


def pickdump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)