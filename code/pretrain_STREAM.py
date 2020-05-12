import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import os
import pprint
import argparse

from cfg.config import cfg, cfg_from_file
from datasets import get_loader
from STREAM.trainer import train, validate
from model import Encoder, Decoder
from process_data import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file', default='cfg/pretrain_STREAM.yml',
                        help='optional config file', type=str)
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--root_data_dir', dest='root_data_dir', type=str, default='')
    parser.add_argument('--preprocess_threshold', dest='threshold', type=int)
    return parser.parse_args()


#############
# Init model
#############
def init_model(vocabulary):

    encoder = Encoder().to(cfg.DEVICE)
    decoder = Decoder(vocab=vocabulary).to(cfg.DEVICE)
    if cfg.TRAIN.CAP_CNN and cfg.TRAIN.CAP_RNN:
        print('Pre-Trained Caption Model')
        encoder_checkpoint = torch.load(cfg.TRAIN.CAP_CNN, map_location=lambda storage, loc: storage)
        decoder_checkpoint = torch.load(cfg.TRAIN.CAP_RNN, map_location=lambda storage, loc: storage)

        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=cfg.TRAIN.DECODER_LR)
        decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
        decoder_optimizer.load_state_dict(decoder_checkpoint['optimizer_state_dict'])
    else:
        decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=cfg.TRAIN.DECODER_LR)

    return encoder, decoder, decoder_optimizer


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

    if cfg.TRAIN.FLAG:
        # Load data
        train_loader = get_loader('train', vocab, cfg.TRAIN.BATCH_SIZE,
                                  transform=transform, norm=norm)
        train(encoder=enc, decoder=dec, decoder_optimizer=dec_optim,
              criterion=crit, train_loader=train_loader)

    if cfg.TRAIN.VALIDATE:
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

    if args.threshold is not None:
        cfg.VOCAB.THRESHOLD = args.threshold

    print('Using config:')
    pprint.pprint(cfg)

if __name__ == '__main__':
    set_config_params()

    caption_path = os.path.join(cfg.DATA_DIR, 'annotations/captions_train2014.json')
    vocab_path = os.path.join(cfg.DATA_DIR, 'vocab.pkl')

    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    pretrain_STREAM()

