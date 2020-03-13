'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py
2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
This script processes the COCO dataset
'''

from cfg.config import cfg

import os
import pickle
from collections import Counter
import nltk
from PIL import Image
from pycocotools.coco import COCO
import argparse
import pprint


def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--root_dir', dest='root_dir', type=str, default='')
    return parser.parse_args()


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

def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # ommit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('<unk>') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def main(caption_path,vocab_path,threshold):
    vocab = build_vocab(json=caption_path,threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("resizing images...")
    splits = ['val','train']

    for split in splits:
        folder = os.path.join(cfg.DATA_DIR, '%s2014' % split)
        resized_folder = os.path.join(cfg.DATA_DIR, '%s2014_resized/' % split)
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, image_file), image.format)

            if i % 1000 == 0 or i == num_images:
                print("copied %s/%s images" % (str(i),str(num_images)))

        print("copied %s images in %s folder..." % (str(num_images), split))

    print("done resizing images...")


if __name__ == '__main__':
    args = parse_args()

    if args.root_dir != '':
        cfg.ROOT_DIR = args.ROOT_DIR
    if args.data_size != '':
        cfg.DATASET_SIZE = args.data_size

    cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, cfg.DATASET_SIZE)
    print('Using config:')
    pprint.pprint(cfg)

    caption_path = os.path.join(cfg.DATA_DIR, 'annotations/captions_train2014.json')
    vocab_path = os.path.join(cfg.DATA_DIR, 'vocab.pkl')
    threshold = 5

    main(caption_path, vocab_path, threshold)
