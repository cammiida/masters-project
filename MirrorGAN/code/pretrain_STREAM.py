import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import os
import pprint
import argparse

from cfg.config import cfg, cfg_from_file
from datasets import get_loader
from STREAM.data_processer import init_model, process_data
from STREAM.trainer import train, validate

from datasets import Vocabulary

def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file', default='cfg/pretrain_STREAM.yml',
                        help='optional config file', type=str)
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--root_data_dir', dest='root_data_dir', type=str, default='')
    parser.add_argument('--preprocess', dest='preprocess_data', type=bool, default=False)
    parser.add_argument('--preprocess_threshold', dest='threshold', type=int)
    return parser.parse_args()


def pretrain_STREAM():
    crit = nn.CrossEntropyLoss().to(cfg.DEVICE)
    enc, dec, dec_optim = init_model(vocab)

    # rasnet tranformation/normalization
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip()
    ])
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    ######################
    # Run training/validation
    ######################

    if cfg.STREAM.TRAIN_MODEL:
        # Load data
        train_loader = get_loader('train', vocab, cfg.TRAIN.BATCH_SIZE,
                                  transform=transform, norm=norm)
        train(encoder=enc, decoder=dec, decoder_optimizer=dec_optim,
              criterion=crit, train_loader=train_loader)

    if cfg.STREAM.VALID_MODEL:
        # Load data
        val_loader = get_loader('val', vocab, cfg.TRAIN.BATCH_SIZE,
                                transform=transform, norm=norm)
        # Don't caluclate gradients for validation
        with torch.no_grad():
            validate(encoder=enc, decoder=dec, criterion=crit, val_loader=val_loader)


def set_config_params():
    args = parse_args()
    if args.cfg_file != '':
        cfg_from_file(args.cfg_file)

    if args.root_data_dir != '':
        cfg.ROOT_DIR = args.ROOT_DATA_DIR
    if args.data_size != '':
        cfg.DATASET_SIZE = args.data_size

    cfg.DATA_DIR = os.path.join(cfg.ROOT_DATA_DIR, cfg.DATASET_SIZE)

    # Set device
    if torch.cuda.is_available():
        cfg.DEVICE = torch.device('cuda')

    if args.preprocess_data is not None:
        cfg.STREAM.PREPROCESS_DATA = args.preprocess_data

    if args.threshold is not None:
        cfg.STREAM.THRESHOLD = args.threshold

    print('Using config:')
    pprint.pprint(cfg)

if __name__ == '__main__':
    set_config_params()

    caption_path = os.path.join(cfg.DATA_DIR, 'annotations/captions_train2014.json')
    vocab_path = os.path.join(cfg.DATA_DIR, 'vocab.pkl')

    if cfg.STREAM.PREPROCESS_DATA:
        process_data(caption_path, vocab_path)

    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    pretrain_STREAM()

