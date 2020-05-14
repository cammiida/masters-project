from __future__ import print_function

from cfg.config import cfg, cfg_from_file
from datasets import get_loader, collate_fn
from process_data import Vocabulary
from trainer import Trainer
from miscc.utils import str2bool

import os
import time
import pickle
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Train a MirrorGAN network')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='', type=str)
    parser.add_argument('--root_data_dir', dest='root_data_dir', type=str, default='')
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    parser.add_argument('--original_STREAM', dest='use_original_STREAM', type=str2bool, default=False,
                        help='Argument for using the original STREAM model in the MirrorGAN paper')
    return parser.parse_args()


# TODO: Fix this so it fits new dataset
def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r", encoding='utf-8') as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r", encoding='utf-8') as f:
                print('Load from:', name)
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


def set_config_params():
    # Get file
    args = parse_args()
    if args.cfg_file != '': cfg_from_file(args.cfg_file)

    # Set config
    if args.root_data_dir != '': cfg.ROOT_DATA_DIR = args.root_data_dir
    if args.data_size != '': cfg.DATASET_SIZE = args.data_size
    cfg.DATA_DIR = os.path.join(cfg.ROOT_DATA_DIR, cfg.DATASET_SIZE)

    # Set seed for train or not train env
    if not cfg.TRAIN.FLAG: args.manual_seed = 100
    elif args.manual_seed is None: args.manual_seed = random.randint(1, 10000)

    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    # Check that models config parameters are set
    assert cfg.MODELS_DIR != '', \
        "Directory for models must be set."
    assert cfg.TRAIN.NET_E != '' and cfg.TRAIN.CAP_CNN != '' and cfg.TRAIN.CAP_RNN != '', \
        "Model names must be specified."

    # Make sure the right values are set if the original STREAM is used
    if args.use_original_STREAM:
        assert cfg.DATASET_SIZE == 'big', \
            "Dataset size must be big to use original STREAM"
        cfg.TREE.BRANCH_NUM = 1
        cfg.TEXT.EMBEDDING_DIM = 256
        cfg.TRAIN.STREAM.USE_ORIGINAL = True
        cfg.TRAIN.STREAM.HIDDEN_SIZE = 512

    # Set STREAM model paths
    if args.use_original_STREAM:
        i_cnn = cfg.TRAIN.CAP_CNN.rfind('/')
        i_rnn = cfg.TRAIN.CAP_RNN.rfind('/')
        cfg.TRAIN.CAP_CNN = os.path.join(cfg.MODELS_DIR, cfg.DATASET_SIZE, cfg.TRAIN.CAP_CNN[:i_cnn], 'original', cfg.TRAIN.CAP_CNN[i_cnn+1:])
        cfg.TRAIN.CAP_RNN = os.path.join(cfg.MODELS_DIR, cfg.DATASET_SIZE, cfg.TRAIN.CAP_RNN[:i_rnn], 'original', cfg.TRAIN.CAP_RNN[i_rnn+1:])
    else:
        cfg.TRAIN.CAP_CNN = os.path.join(cfg.MODELS_DIR, cfg.DATASET_SIZE, cfg.TRAIN.CAP_CNN)
        cfg.TRAIN.CAP_RNN = os.path.join(cfg.MODELS_DIR, cfg.DATASET_SIZE, cfg.TRAIN.CAP_RNN)

    # Set Generator and STEM model paths
    if cfg.TRAIN.NET_G != '':
        cfg.TRAIN.NET_G = os.path.join(cfg.MODELS_DIR, cfg.DATASET_SIZE, cfg.TRAIN.NET_G)
    cfg.TRAIN.NET_E = os.path.join(cfg.MODELS_DIR, cfg.DATASET_SIZE, cfg.TRAIN.NET_E)

    # Set device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manual_seed)
        cfg.DEVICE = torch.device('cuda')

    print('Using config:')
    pprint.pprint(cfg)


if __name__ == '__main__':
    set_config_params()

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/output/%s/%s_%s_%s' % \
                 (cfg.OUTPUT_PATH, cfg.DATASET_SIZE,
                  cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    print('output_dir: ', output_dir)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bsuffle = False
        split_dir = 'valid'

    # Load vocabulary
    f = open(os.path.join(cfg.DATA_DIR, 'vocab.pkl'), 'rb')
    vocab = pickle.load(f)

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    # rasnet transformation/normalization
    transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])

    ######################
    # Run training/validation
    ######################


    # Load data
    train_loader = get_loader('train', vocab, cfg.TRAIN.BATCH_SIZE, transform=transform)
    # train(encoder=enc, decoder=dec, decoder_optimizer=dec_optim,
    #      criterion=crit, train_loader=train_loader)
    algo = Trainer(output_dir, train_loader, vocab=vocab)
    start_t = time.time()

    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)
        else:
            gen_example(vocab.word2idx, algo) # generate images for customized captions

    end_t = time.time()
    print('Total time for trainig: ', end_t - start_t)






