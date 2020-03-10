'''
This code was written by following the following tutorial:
Link: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
This script processes and generates GloVe embeddings
'''
# coding: utf-8

import pickle
import numpy as np
import bcolz
import os
from processData import Vocabulary


def create_glove_files(root_dir):

    words = []
    idx = 0
    word2idx = {}

    vectors = bcolz.carray(np.zeros(1), rootdir=os.path.join(root_dir, '6B.300.dat'), mode='w')

    with open(os.path.join(root_dir, 'glove.6B.300d.txt'), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=os.path.join(root_dir, '6B.300.dat'), mode='w')
    vectors.flush()
    pickle.dump(words, open(os.path.join(root_dir, '6B.300_words.pkl'), 'wb'))
    pickle.dump(word2idx, open(os.path.join(root_dir, '6B.300_idx.pkl'), 'wb'))


def load_glove_files(root_dir):
    print('Loading GloVe files...')
    vectors = bcolz.open(os.path.join(root_dir, '6B.300.dat'))[:]
    words = pickle.load(open(os.path.join(root_dir, '6B.300_words.pkl'), 'rb'))
    word2idx = pickle.load(open(os.path.join(root_dir, '6B.300_idx.pkl'), 'rb'))

    print('GloVe files are loaded')

    return vectors, words, word2idx


def generate_weight_matrix(root_dir, glove_dir, vectors, words, word2idx):

    print('Loading vocab...')
    with open(os.path.join(root_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    print('Vocab loaded')

    glove = {w: vectors[word2idx[w]] for w in words}
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for i, word in enumerate(vocab.idx2word):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))

    print('Creating weights_matrix...')

    pickle.dump(weights_matrix, open(os.path.join(glove_dir, 'glove_words.pkl'), 'wb'), protocol=2)

    print('weights_matrix is created')


if __name__ == '__main__':
    # TODO: Get paths from config
    data_dir = '../data/'
    glove_dir = os.path.join(data_dir, 'glove.6B/')

    create_glove_files(glove_dir)
    vectors, words, word2idx = load_glove_files(glove_dir)
    generate_weight_matrix(data_dir, glove_dir, vectors, words, word2idx)
