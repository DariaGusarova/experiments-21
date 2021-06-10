import numpy as np
from copy import deepcopy
from tqdm import tqdm_notebook


def to_one_hot(batch, mask, lang):
    # batch.shape - (batch_size, length)
    batch_size, length = len(batch), len(batch[0])
    batch_one_hot = np.zeros((batch_size, length, len(lang)))
    batch_idx = np.zeros((batch_size, length))
    mask_mod = deepcopy(mask)
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


def separate_dataset(path, subpath, batch_size, load_epoches):
    print('Loading dataset ...')
    en_sent, de_sent, en_lens, de_lens = read_files(path + subpath + '.en', path + subpath + '.de')
    print('Dataset loaded.')
    print('Shuffling dataset ...')
    en_sent, de_sent, en_lens, de_lens = shuffle(en_sent, de_sent, en_lens, de_lens)
    print('Dataset shuffled.')
    print('Shuffling batches ...')
    batch_ix = np.arange(0, len(en_sent)-batch_size, batch_size)
    np.random.shuffle(batch_ix)
    print('Batches shuffled.')
    folder = path + subpath
    print('Writing separated dataset ...')
    for i in tqdm_notebook(range(len(batch_ix))):
        idx = batch_ix[i]
        if (i % load_epoches == 0):
            curr_file = folder + '/' + str(int(i // load_epoches))
            if (i != 0):
                f_en.close()
                f_de.close()
            f_en = open(curr_file + '.en', 'w')
            f_de = open(curr_file + '.de', 'w')
        for j in range(batch_size):
            print(str(en_lens[idx+j]) + " " + " ".join(en_sent[idx+j]), file=f_en)
            print(str(de_lens[idx+j]) + " " + " ".join(de_sent[idx+j]), file=f_de)
    f_en.close()
    f_de.close()
    print('Separated dataset was written.')
    return 


def extract_stats(file, bd=6., gap=10):
    f = open(file, 'r')
    sm_entropy, entropy, accuracy, times = [], [], [], []
    for line in f:
        curr = line.split()
        sm_entropy.append(float(curr[4]))
        entropy.append(float(curr[6][:-1]))
        accuracy.append(float(curr[8][:-1]))
        times.append(float(curr[10]))
    f.close()
    times = np.array(times) 
    for i in range(len(times)):
        if times[i] < bd:
            times[i] = np.mean(times[i-gap:i])/2. + np.mean(times[i+1:i+1+gap])/2.
    return {'sm_entropy': np.array(sm_entropy), 'entropy': np.array(entropy), 
            'accuracy': np.array(accuracy), 'time': times}
