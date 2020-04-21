from datasets import get_loader, Vocabulary
from trainer import Trainer
from model import CNN_ENCODER

import os
import pickle

import torchvision.transforms as transforms
import torch

def prepare_imgs(data):
    imgs, captions, captions_lens = data
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    real_imgs = [0]
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]

    return imgs


batch_size = 64
tree_base_size = 299
tree_branch_num = 1
data_dir = '../../data/small'
text_embedding_dim = 256

imsize = tree_base_size * (2 ** (tree_branch_num - 1))
# rasnet transformation/normalizations
transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])
cnn_model = CNN_ENCODER(text_embedding_dim)

# Load vocabulary
f = open(os.path.join(data_dir, 'vocab.pkl'), 'rb')
vocab = pickle.load(f)

train_loader = get_loader('train', vocab, batch_size, transform=transform, root_dir=data_dir)

print("vocab:", vocab)
print("len vocab: ", len(vocab))
print("train_loader: ", train_loader)

# train
max_i = 1
for i, data in enumerate(train_loader):
    #cnn_model.zero_grad()
    if i >= max_i:
        break

    imgs, captions, cap_lens = data
    print("cap_lens: ", cap_lens)
    imgs = prepare_imgs(data)
    print("imgs: ", imgs)

    print("imgs[-1]: ", imgs[-1])
    print("imgs[-1].size(): ", imgs[-1].size())

    print("Sending imgs to cnn_model...")
    words_feartures, sent_code = cnn_model(imgs[-1])




