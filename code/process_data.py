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
import argparse
import pprint
import nltk

from collections import Counter
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--root_data_dir', dest='root_data_dir', type=str, default='')
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--threshold', dest='threshold', type=int)
    return parser.parse_args()


def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(tqdm(ids)):
        caption = str(coco.anns[id]['caption'])
        # TODO: Can this be done with BERT tokenizer?
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokens = tokenizer.tokenize(caption.lower())
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # omit non-frequent words
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
    # Make square based on the smallest dim
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
    # Downsample with high quality (antialias)
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image


def process_data(caption_path, vocab_path, threshold):
    # Create vocab
    print("Building vocab...")
    vocab = build_vocab(json=caption_path, threshold=threshold)
    # Save vocab
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Saved vocab at ", vocab_path)

    # Resize images
    print("Resizing images...")
    splits = ['val', 'train']

    for split in splits:
        folder = os.path.join(cfg.DATA_DIR, '%s2014' % split)
        resized_folder = os.path.join(cfg.DATA_DIR, '%s2014_resized/' % split)
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(tqdm(image_files)):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, image_file), image.format)

            if i % 1000 == 0 or i == num_images:
                print("Copied %s/%s images" % (str(i), str(num_images)))

        print("Copied %s images in %s folder..." % (str(num_images), split))

    print("Done resizing images.")


if __name__ == '__main__':
    args = parse_args()
    if args.root_data_dir != '': cfg.ROOT_DATA_DIR = args.root_data_dir
    if args.data_size != '': cfg.DATASET_SIZE = args.data_size
    cfg.DATA_DIR = os.path.join(cfg.ROOT_DATA_DIR, cfg.DATASET_SIZE)
    if args.threshold is not None: cfg.VOCAB.THRESHOLD = args.threshold

    print("Using config:")
    pprint.pprint(cfg)

    caption_path = os.path.join(cfg.DATA_DIR, 'annotations/captions_train2014.json')
    vocab_path = os.path.join(cfg.DATA_DIR, 'vocab.pkl')

    process_data(caption_path, vocab_path, threshold=cfg.VOCAB.THRESHOLD)


