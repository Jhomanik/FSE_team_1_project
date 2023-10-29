from collections import Counter
import sys
from tqdm import tqdm
import numpy as np
import h5py
import json
from torch.utils.data import Dataset

class IdxDataset(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        return self.dataset[self.idx[idx]]

def create_dict(captions_tokenized_file):
    with open(captions_tokenized_file) as f:
        captions = json.load(f)
    
    word_counts = Counter()
    
    for img_i in range(len(captions)):
        for caption_i in range(len(captions[img_i])):
            sentence = captions[img_i][caption_i]
            captions[img_i][caption_i] = ["#START#"]+sentence.split(' ')+["#END#"]
    
    for img_i in range(len(captions)):
        for caption_i in range(len(captions[img_i])):
            sentence = captions[img_i][caption_i]
            word_counts.update(sentence[1:-1])
    vocab  = ['#UNK#', '#START#', '#END#', '#PAD#']
    vocab += [k for k, v in word_counts.items() if v >= 5 if k not in vocab]
    
    word_to_index = {w: i for i, w in enumerate(vocab)}
    return word_to_index, vocab
       
def create_dataset(img_file, captions_tokenized_file):
    f = h5py.File(img_file, 'r')
    img_codes = f['data']
    
    word_to_index, vocab = create_dict(captions_tokenized_file)
    captions = np.array(captions)

    np.random.seed(42)
    perm = np.random.permutation(len(img_codes))
    threshold = round(len(img_codes) * 0.1)
    train_img_idx, val_img_idx = perm[threshold:], perm[: threshold]

    train_img_idx.sort()
    val_img_idx.sort()
    train_img_codes = IdxDataset(img_codes, train_img_idx)
    val_img_codes = IdxDataset(img_codes, val_img_idx)
    train_captions = IdxDataset(captions, train_img_idx)
    val_captions = IdxDataset(captions, val_img_idx)

    return [train_img_codes, train_captions] , [val_img_codes, val_captions], word_to_index, vocab