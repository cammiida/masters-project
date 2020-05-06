from STREAM.data_processer import build_vocab
from cfg.config import cfg

import os
import argparse
import pprint
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--root_data_dir', dest='root_data_dir', type=str, default='')
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--threshold', dest='threshold', type=int)
    return parser.parse_args()


def save_vocab(vocab, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)


def set_config_params():
    args = parse_args()

    if args.root_data_dir != '': cfg.ROOT_DATA_DIR = args.root_data_dir
    if args.data_size != '': cfg.DATASET_SIZE = args.data_size
    cfg.DATA_DIR = os.path.join(cfg.ROOT_DATA_DIR, cfg.DATASET_SIZE)

    if args.threshold is not None: cfg.VOCAB.THRESHOLD = args.threshold

    print("Using config:")
    pprint.pprint(cfg)


if __name__ == '__main__':
    set_config_params()

    caption_path = os.path.join(cfg.DATA_DIR, 'annotations/captions_train2014.json')
    vocab_path = os.path.join(cfg.DATA_DIR, 'vocab_old.pkl')

    vocab = build_vocab(caption_path)
    print("len vocab: ", len(vocab))
    save_vocab(vocab, vocab_path)

