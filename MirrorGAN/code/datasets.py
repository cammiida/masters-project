from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from cfg.config import cfg

import torch
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import nltk
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# TODO: Do we even need this since collate_fn is used?
'''
def prepare_data(data):
    print("prepare data")
    #imgs, captions, captions_lens, class_ids, keys = data
    imgs, captions, captions_lens = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    print("imgs: ", imgs)
    print("sorted_cap_indices:", sorted_cap_indices)
    print("prepare data, imgs size: ", imgs.size())
    print("sorted_cap_indices size: ", sorted_cap_indices.size())
    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        real_imgs.append(imgs[i]).to(cfg.DEVICE)

    print("captions size before squeeze: ", captions.size())
    captions = captions[sorted_cap_indices].squeeze()
    print("captions size after squeeze: ", captions.size())
    #class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    #keys = [keys[i] for i in sorted_cap_indices.numpy()]

    # Were Variables
    captions = captions.to(cfg.DEVICE)
    sorted_cap_lens = sorted_cap_lens.to(cfg.DEVICE)

    return [real_imgs, captions, sorted_cap_lens] #, class_ids, keys]
'''


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    # ret contains an array with a one or more tensors (images) inside.
    return ret




class DataLoader(torch.utils.data.Dataset):
    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        # TODO: Check if normalize should be like in MirrorGAN?
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.imsize = []
        base_size = cfg.TREE.BASE_SIZE
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size *= 2


    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        img_path = os.path.join(self.root, path)
        image = get_imgs(img_path, self.imsize,
                         transform=self.transform, normalize=self.norm)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = list()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    # Sort batch by length of captions
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    # create separate tuples of images and captions from batch
    images, captions = zip(*batch)

    new_images = []
    for j in range(len(images[0])):
        img_list = []
        for i in range(len(images)):
            img_list.append(images[i][j])
        new_images.append(torch.stack(img_list))
    images = new_images

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths


def get_loader(method, vocab, batch_size, transform):
    root_dir = cfg.DATA_DIR
    root = None
    json = None
    # train/validation paths
    if method == 'train':
        root = os.path.join(root_dir, 'train2014_resized')
        json = os.path.join(root_dir, 'annotations/captions_train2014.json')
    elif method == 'val':
        root = os.path.join(root_dir, 'val2014_resized')
        json = os.path.join(root_dir, 'annotations/captions_val2014.json')


    coco = DataLoader(root=root, json=json, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              collate_fn=collate_fn)

    return data_loader


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)









