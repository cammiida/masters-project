import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import os
import pprint
import argparse
from datetime import datetime
import dateutil.tz

from cfg.config import cfg, cfg_from_file
from datasets import get_loader
from STREAM.trainer import train, validate
from model import Encoder, Decoder
from process_data import Vocabulary
from miscc.utils import mkdir_p, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file', default='cfg/pretrain_STREAM.yml',
                        help='optional config file', type=str)
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--root_data_dir', dest='root_data_dir', type=str, default='')
    parser.add_argument('--train', dest='train', type=str2bool)
    parser.add_argument('--val', dest='validate', type=str2bool)
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

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/output/%s/%s_%s_%s' % \
                 (cfg.OUTPUT_PATH, cfg.DATASET_SIZE,
                  cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    mkdir_p(output_dir)

    ######################
    # Run training/validation
    ######################

    if cfg.TRAIN.FLAG:
        # Load data
        train_loader = get_loader('train', vocab, cfg.TRAIN.BATCH_SIZE,
                                  transform=transform, norm=norm)
        train(encoder=enc, decoder=dec, decoder_optimizer=dec_optim,
              criterion=crit, train_loader=train_loader, output_dir=output_dir)

    if cfg.TRAIN.VALIDATE:
        # Load data
        val_loader = get_loader('val', vocab, cfg.TRAIN.BATCH_SIZE,
                                transform=transform, norm=norm)
        # Don't caluclate gradients for validation
        with torch.no_grad():
            validate(encoder=enc, decoder=dec, criterion=crit,
                     val_loader=val_loader, vocab=vocab, output_dir=output_dir)


def set_config_params():
    args = parse_args()
    if args.cfg_file != '':
        cfg_from_file(args.cfg_file)

    if args.train is not None:
        print("args.train: ", args.train)
        cfg.TRAIN.FLAG = args.train
    if args.validate is not None:
        print("args.validate: ", args.validate)
        cfg.TRAIN.VALIDATE = args.validate

    assert cfg.TRAIN.FLAG or cfg.TRAIN.VALIDATE, \
        "Must either train or validate to run"

    if args.root_data_dir != '':
        cfg.ROOT_DIR = args.ROOT_DATA_DIR
    if args.data_size != '':
        cfg.DATASET_SIZE = args.data_size

    cfg.DATA_DIR = os.path.join(cfg.ROOT_DATA_DIR, cfg.DATASET_SIZE)
    # If model names are not empty...
    if cfg.TRAIN.CAP_CNN and cfg.TRAIN.CAP_RNN:
        assert cfg.MODELS_DIR != '', \
            "Directory for models must be set."

        cfg.TRAIN.CAP_CNN = os.path.join(cfg.MODELS_DIR, cfg.DATASET_SIZE, cfg.TRAIN.CAP_CNN)
        cfg.TRAIN.CAP_RNN = os.path.join(cfg.MODELS_DIR, cfg.DATASET_SIZE, cfg.TRAIN.CAP_RNN)

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
    vocab_path = os.path.join(cfg.DATA_DIR, 'vocab_old.pkl')

    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    pretrain_STREAM()

