import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import os
import pprint
import argparse

from cfg.config import cfg
from datasets import get_loader, STREAM_collate_fn
from STREAM.data_processer import init_model, process_data
from STREAM.trainer import train, validate


def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--root_dir', dest='root_dir', type=str, default='')
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

    if cfg.TRAIN_MODEL:
        # Load data
        train_loader = get_loader('train', vocab, cfg.BATCH_SIZE,
                                  transform=transform, norm=norm, collate_fn=STREAM_collate_fn)
        train(encoder=enc, decoder=dec, decoder_optimizer=dec_optim,
              criterion=crit, train_loader=train_loader)

    if cfg.VALID_MODEL:
        # Load data
        val_loader = get_loader('val', vocab, cfg.BATCH_SIZE,
                                transform=transform, norm=norm, collate_fn=STREAM_collate_fn)
        # Don't caluclate gradients for validation
        with torch.no_grad():
            validate(encoder=enc, decoder=dec, criterion=crit, val_loader=val_loader)


def set_config_params():
    args = parse_args()

    if args.root_dir != '':
        cfg.ROOT_DIR = args.ROOT_DIR
    if args.data_size != '':
        cfg.DATASET_SIZE = args.data_size

    cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, cfg.DATASET_SIZE)
    print('Using config:')
    pprint.pprint(cfg)

    # Set device
    if torch.cuda.is_available():
        cfg.DEVICE = torch.device('cuda')

if __name__ == '__main__':
    set_config_params()

    # Load vocabulary
    with open(os.path.join(cfg.DATA_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    caption_path = os.path.join(cfg.DATA_DIR, 'annotations/captions_train2014.json')
    vocab_path = os.path.join(cfg.DATA_DIR, 'vocab.pkl')
    threshold = 5

    process_data(caption_path, vocab_path, threshold)

    pretrain_STREAM()

