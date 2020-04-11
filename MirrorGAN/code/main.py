from __future__ import print_function

from cfg.config import cfg, cfg_from_file
from datasets import DataLoader, get_loader
from trainer import Trainer as trainer

import os
import sys
import time
import pickle
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train a MirrorGAN network')
    # TODO: Add default config file
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/...', type=str)
    parser.add_argument('--root_data_dir', dest='root_data_dir', type=str, default='')
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def main():
    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    # rasnet transformation/normalization
    transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    ######################
    # Run training/validation
    ######################

    if cfg.TRAIN_MODEL:
        # Load data
        train_loader = get_loader('train', vocab, cfg.BATCH_SIZE, root_dir=cfg.DATA_DIR)
        #train(encoder=enc, decoder=dec, decoder_optimizer=dec_optim,
        #      criterion=crit, train_loader=train_loader)
        # TODO: Fix that trainer doesn't take the dataset.n_words or dataset.ixtoword arguments
        algo = trainer(output_dir, train_loader, vocab=vocab)
        train_start_t = time.time()
        algo.train()
        train_end_t = time.time()
        print('Total time for trainig: ', train_end_t - train_start_t)

    if cfg.VALID_MODEL:
        # Load data
        val_loader = get_loader('val', vocab, cfg.BATCH_SIZE, root_dir=cfg.DATA_DIR)
        # Don't caluclate gradients for validation
        #with torch.no_grad():
        #    validate(encoder=enc, decoder=dec, criterion=crit, val_loader=val_loader)
        algo = trainer(output_dir, val_loader, vocab)
        val_start_t = time.time()
        algo.sampling(split_dir)
        val_end_t = time.time()
        print('Total time for validation: ', val_end_t - val_start_t)

        # TODO: Add possibility to generate images for customized captions
        # like in MirrorGAN main.py


if __name__ == '__main__':
    # Get file
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # Set config
    if args.root_data_dir != '':
        cfg.ROOT_DATA_DIR = args.root_data_dir
    if args.data_size != '':
        cfg.DATASET_SIZE = args.data_size

    cfg.DATA_DIR = os.path.join(cfg.ROOT_DATA_DIR, cfg.DATASET_SIZE)
    print('Using config:')
    pprint.pprint(cfg)

    # Set seed for train or not train env
    if not cfg.TRAIN.FLAG:
        args.manual_seed = 100
    elif args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    # Set device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manual_seed)
        cfg.DEVICE = torch.device('cuda')
    else:
        cfg.DEVICE = torch.device('cpu')

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/output/%s_%s_%s' % \
                 (cfg.OUTPUT_PATH, cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    print('output_dir: ', output_dir)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bsuffle = False
        split_dir = 'valid'

    # Load vocabulary
    with open(os.path.join(cfg.DATA_DIR, 'vocab_pkl'), 'rb') as f:
        vocab = pickle.load(f)


    main()





