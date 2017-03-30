''' Loading various datasets
'''
from pathlib import Path
import numpy as np
import requests
import pickle

# class DataSet(object):
#     def __init__(self):
#         pass

#     def next_batch(self, size: int):
#         ''' Return next batch of given @size
#         '''
#         pass

#     def load(self, data_fpath: Path):

root_fpath = Path.home() / 'local' / 'data'
data_fpath = root_fpath / 'cifar' / 'cifar-10-batches-py'

# Download the files if we don't have them already
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#if not data_fpath.exists():
req = requests.get(url, stream=True)
tarball_fpath = root_fpath / 'cifar' / 'cifar-10-python.tar.gz'
# http://docs.python-requests.org/en/latest/user/advanced/#body-content-workflow
with tarball_fpath.open('wb') as fout:
    for chunk in req.iter_content(chunk_size=1024):
        if not chunk: continue
        fout.write(chunk)    
        
assert data_fpath.exists()
batch_fps = data_fpath.glob('data_batch_*')
assert batch_fps

feats_list = []
labels_list = []
for fp in batch_fps:
    print(str(fp))
    with fp.open('rb') as fin:
        # The data is stored pickled numpy array
        # Use the following settings to conform to original encoding
        _st = pickle.load(fin, 
                          fix_imports=True,
                          encoding='bytes')
        feats_list.append(_st[b'data'])
        labels_list.append(_st[b'labels'])

feats = np.concatenate(feats_list)
labels = np.concatenate(labels_list)
assert 3072 == labels.shape[1], 'feature dimension must match'
