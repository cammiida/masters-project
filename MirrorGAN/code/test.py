from datasets import get_loader, Vocabulary
from trainer import Trainer
from model import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence

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

from miscc.losses import caption_loss



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


caption_rnn = Decoder(vocab)
caption_cnn = Encoder()

rnn_checkpoint = torch.load('../../models/STREAM/small/rnn_decoder', map_location=lambda storage, log: storage)
caption_rnn.load_state_dict(rnn_checkpoint['model_state_dict'])

cnn_checkpoint = torch.load('../../models/STREAM/small/cnn_encoder', map_location=lambda storage, loc: storage)
caption_cnn.load_state_dict(cnn_checkpoint['model_state_dict'])



train_loader = get_loader('train', vocab, batch_size,
                          transform=transform)


device = torch.device('cpu')
real_batch = next(iter(train_loader))

real_images = real_batch[0]
sizes = [imgs.size() for imgs in real_images]

print(sizes)
'''
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training images")
show_images = real_images[2]
print(np.transpose(vutils.make_grid(show_images[:64], padding=2, normalize=True), (1,2,0)).shape)
print(show_images.size())
plt.imshow(np.transpose(vutils.make_grid(show_images.to(device)[:64], padding=2,
                                         normalize=True), (1,2,0)))

plt.show()
'''
# train
max_i = 1
for i, data in enumerate(tqdm(train_loader)):
    print("i: ", i)

    if i >= max_i:
        break

    caption_rnn.zero_grad()
    caption_cnn.zero_grad()

    imgs, captions, cap_lens = data


    fakeimg_feature = caption_cnn(imgs[i])
    if isinstance(cap_lens, torch.Tensor):
        cap_lens = cap_lens.data.tolist()


    scores, caps_sorted, decode_lengths, alphas = caption_rnn(fakeimg_feature, captions, cap_lens)
    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

    targets = caps_sorted[:, 1:]
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]


    cap_loss = caption_loss(scores, targets) * cfg.TRAIN.SMOOTH.LAMBDA1


    print("targets: ", targets)
    print("targets size: ", targets.size())


    print("cap_loss: ", cap_loss)

    if i < len(train_loader): # skipping last batch if batch size is not a multiple of
        #print("captions: ", captions)
        #print("cap_lens: ", cap_lens)
        img_sizes = [img.size() for img in imgs]
        print("img_sizes: ", img_sizes)
        #print("initializing rnn hidden")
        #hidden = rnn_model.init_hidden(batch_size)
        #print("rnn forward")
        #words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        #print("Sending imgs to cnn_model...")
        #words_feartures, sent_code = cnn_model(imgs)





