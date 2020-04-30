import os
import pickle

import torch

from cfg.config import cfg
from model import CNN_ENCODER, RNN_ENCODER


# Load vocabulary
f = open(os.path.join(cfg.DATA_DIR, 'vocab.pkl'), 'rb')
vocab = pickle.load(f)

n_words = len(vocab)


image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
print('image_encoder: ', image_encoder)
img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
state_dict = \
    torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
image_encoder.load_state_dict(state_dict)
for p in image_encoder.parameters():
    p.requires_grad = False
print('Load image encoder from: ', img_encoder_path)
image_encoder.eval()

text_encoder = \
    RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
state_dict = \
    torch.load(cfg.TRAIN.NET_E,
               map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder from: ', cfg.TRAIN.NET_E)
text_encoder.eval()