from datasets import get_loader
from datasets import Vocabulary, collate_fn
from trainer import Trainer
from model import Encoder, Decoder, CAPTION_CNN, CAPTION_RNN
from cfg.config import cfg, cfg_from_file
from miscc.losses import caption_loss

import os
import pickle

from tqdm import tqdm

import torchvision.transforms as transforms
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

#cfg_from_file('./cfg/pretrain_DAMSM.yml')
cfg.DATA_DIR = '../../data/small'


batch_size = cfg.TRAIN.BATCH_SIZE
tree_base_size = cfg.TREE.BASE_SIZE # 299
tree_branch_num = cfg.TREE.BRANCH_NUM
text_embedding_dim = 256
load_old_stream = False

imsize = tree_base_size * (2 ** (tree_branch_num - 1))
# rasnet transformation/normalizations
transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()
])


# only with big dataset

if load_old_stream:
    f = open('../../models/big/original_STREAM/vocab.pkl', 'rb')
    vocab = pickle.load(f)

    caption_cnn = CAPTION_CNN(embed_size=cfg.STREAM.EMBED_SIZE)
    caption_rnn = CAPTION_RNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1)

    cnn_checkpoint = torch.load('../../models/big/original_STREAM/cnn_encoder.pkl', map_location=lambda storage, loc: storage)
    caption_cnn.load_state_dict(cnn_checkpoint)
    rnn_checkpoint = torch.load('../../models/big/original_STREAM/rnn_decoder.pkl', map_location=lambda storage, loc: storage)
    caption_rnn.load_state_dict(rnn_checkpoint)
else:
    # Load vocabulary
    f = open(os.path.join(cfg.DATA_DIR, 'vocab.pkl'), 'rb')
    vocab = pickle.load(f)

    caption_rnn = Decoder(vocab)
    caption_cnn = Encoder()

    cnn_checkpoint = torch.load('../../models/small/STREAM/cnn_encoder', map_location=lambda storage, loc: storage)
    caption_cnn.load_state_dict(cnn_checkpoint['model_state_dict'])
    rnn_checkpoint = torch.load('../../models/small/STREAM/rnn_decoder', map_location=lambda storage, log: storage)
    caption_rnn.load_state_dict(rnn_checkpoint['model_state_dict'])



print("len vocab: ", len(vocab))

train_loader = get_loader('train', vocab, batch_size, transform=transform)


device = torch.device('cpu')
real_batch = next(iter(train_loader))

real_images = real_batch[0]
sizes = [imgs.size() for imgs in real_images]

print(sizes)

# train
max_i = 1
for i, data in enumerate(tqdm(train_loader)):
    if i >= max_i:
        break
    print("i: ", i)


    caption_rnn.zero_grad()
    caption_cnn.zero_grad()

    imgs, captions, cap_lens = data
    print("data type: ", type(data))
    print("imgs type: ", type(imgs))
    print("len imgs: ", len(imgs))
    #print("captions: ", captions)


    fakeimg_feature = caption_cnn(imgs[-1])
    if isinstance(cap_lens, torch.Tensor):
        cap_lens = cap_lens.data.tolist()

    # targets = caps_sorted[:, 1:]
    # targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

    if load_old_stream:
        targets = pack_padded_sequence(captions, cap_lens, batch_first=True)[0]
        cap_output = caption_rnn(fakeimg_feature, captions, cap_lens)
        print("cap_output: ", type(cap_output), cap_output.shape)
        print("targets: ", type(targets), targets.shape)
        cap_loss = caption_loss(cap_output, targets) * cfg.TRAIN.SMOOTH.LAMBDA1
    else:
        scores, caps_sorted, decode_lengths, alphas = caption_rnn(fakeimg_feature, captions, cap_lens)
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

        targets = caps_sorted[:, 1:]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        cap_loss = caption_loss(scores, targets) * cfg.TRAIN.SMOOTH.LAMBDA1


        '''cap_output, caps_sorted, decode_lengths, alphas = caption_rnn(fakeimg_feature, captions, cap_lens)
        print("Decode_lengths: ", decode_lengths)
        print("Cap_lens: ", cap_lens)
        print("cap_output before pack_padded_sequence: ", cap_output.shape)
        cap_output = pack_padded_sequence(cap_output, cap_lens, batch_first=True)[0]
        '''

    print("cap_loss: ", cap_loss)







