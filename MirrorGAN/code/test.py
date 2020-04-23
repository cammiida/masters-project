from datasets import get_loader, Vocabulary
from trainer import Trainer
from model import CNN_ENCODER, RNN_ENCODER, Encoder, Decoder


import os
import pickle

from tqdm import tqdm

import torchvision.transforms as transforms
import torch
import numpy as np
from torch.autograd import Variable
from cfg.config import cfg, cfg_from_file

import torchvision.utils as vutils
import matplotlib.pyplot as plt



#cfg_from_file('./cfg/pretrain_DAMSM.yml')
cfg.DATA_DIR = '../../data/small'


#batch_size = 64
batch_size = cfg.TRAIN.BATCH_SIZE
tree_base_size = cfg.TREE.BASE_SIZE # 299
tree_branch_num = cfg.TREE.BRANCH_NUM
text_embedding_dim = 256

imsize = tree_base_size * (2 ** (tree_branch_num - 1))
# rasnet transformation/normalizations
transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()
])


# Load vocabulary
f = open(os.path.join(cfg.DATA_DIR, 'vocab.pkl'), 'rb')
vocab = pickle.load(f)

rnn_model = Decoder(vocab)
#cnn_model = Encoder()
#print("cnn_model: ", cnn_model)
#encoder_checkpoint = torch.load('../../models/STREAM/cnn_encoder', map_location=lambda storage, loc: storage)
#print('encoder_checkpoint: ', encoder_checkpoint['model_state_dict'])
#cnn_model.load_state_dict(encoder_checkpoint['model_state_dict'])
#print("cnn_model: ", cnn_model)

train_loader = get_loader('train', vocab, batch_size,
                          transform=transform)



device = torch.device('cpu')
real_batch = next(iter(train_loader))

real_images = real_batch[0]
sizes = [imgs.size() for imgs in real_images]

print(sizes)

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training images")
show_images = real_images[2]
print(np.transpose(vutils.make_grid(show_images[:64], padding=2, normalize=True), (1,2,0)).shape)
print(show_images.size())
plt.imshow(np.transpose(vutils.make_grid(show_images.to(device)[:64], padding=2,
                                         normalize=True), (1,2,0)))

plt.show()
# train
max_i = 0
for i, data in enumerate(tqdm(train_loader)):
    print("i: ", i)
    if i == len(train_loader):
        print("LAST BATCH")

    #cnn_model.zero_grad()
    rnn_model.zero_grad()
    if i >= max_i:
        break

    imgs, captions, cap_lens = data
    print("len imgs", len(imgs))
    if i < len(train_loader): # skipping last batch if batch size is not a multiple of
        #print("captions: ", captions)
        #print("cap_lens: ", cap_lens)
        lens = [img.size() for img in imgs]
        print(lens)
        #print("initializing rnn hidden")
        hidden = rnn_model.init_hidden(batch_size)
        print("rnn forward")
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
        print("imgs.size(): ", imgs.size())
        #print("imgs[-1].size(): ", imgs[-1].size())

        print("Sending imgs to cnn_model...")
        #words_feartures, sent_code = cnn_model(imgs)





