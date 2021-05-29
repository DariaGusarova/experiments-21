import numpy as np
from copy import deepcopy


def to_one_hot(batch, mask, lang):
    # batch.shape - (batch_size, length)
    batch_size, length = len(batch), len(batch[0])
    batch_one_hot = np.zeros((batch_size, length, len(lang)))
    batch_idx = np.zeros((batch_size, length))
    mask_mod = mask
    for i in range(batch_size):
        for j in range(length):
            idx = lang[batch[i][j]]
            batch_one_hot[i, j, idx] += 1
            batch_idx[i, j] = idx
            if batch[i][j] == '<unk>':
                mask_mod[i, j] = 0.
    return batch_one_hot, batch_idx, mask_mod 


def get_batch(sent, lens):
    max_len = np.max(lens)
    batch = []
    mask = np.ones((len(sent), max_len))
    for i in range(len(sent)):
        curr = deepcopy(sent[i])
        for j in range(max_len - lens[i]):
            curr.append('<end>')
            mask[i, lens[i] + j] = 0
        batch.append(curr)
    return batch, mask 


def read_file(file):
    f = open(file, 'r')
    sent, lens = [], []
    for line in f:
        curr = line.split()
        sent.append(curr[1:])
        lens.append(int(curr[0]))
    f.close()
    sent = np.array(sent)
    lens = np.array(lens)
    return sent, lens


def read_files(file_en, file_de, bound=2):
    en_sent, en_lens = read_file(file_en)
    de_sent, de_lens = read_file(file_de)
    mask = (en_lens > bound) & (de_lens > bound)
    en_sent, en_lens = en_sent[mask], en_lens[mask]
    de_sent, de_lens = de_sent[mask], de_lens[mask]
    return en_sent, de_sent, en_lens, de_lens


def shuffle(en_sent, de_sent, en_lens, de_lens):
    inds = np.arange(len(en_lens))
    np.random.shuffle(inds)
    
    en_lens = en_lens[inds]
    de_lens = de_lens[inds]
    en_sent = en_sent[inds]
    de_sent = de_sent[inds]
    
    order = np.argsort(en_lens + de_lens)
    en_lens = en_lens[order]
    de_lens = de_lens[order]
    en_sent = en_sent[order]
    de_sent = de_sent[order]
    return en_sent, de_sent, en_lens, de_lens
