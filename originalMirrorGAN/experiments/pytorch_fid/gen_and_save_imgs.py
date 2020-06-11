import os
import pickle
import argparse

from datasets import get_loader
from process_data import Vocabulary
from experiments.experimenter import Experimenter
from miscc.utils import mkdir_p

import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision.utils as vutils
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images from validation set')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='../output/experiments/FID_images')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='../data/')
    parser.add_argument('--models_dir', dest='models_dir', type=str, default='../models')
    parser.add_argument('--data_size', dest='data_size', type=str, default='big')
    parser.add_argument('--model_version', dest='model_version', type=str, default='new')
    parser.add_argument('--epoch', dest='epoch', type=int, default=50)
    parser.add_argument('--ref_imgs_dir', dest='ref_imgs_dir', type=str, default='../data/big/val2014_resized')
    return parser.parse_args()


def save_images(save_dir, images):
    if not os.path.isdir(save_dir):
        mkdir_p(save_dir)

    for image in images:
        img_num = len(os.listdir(save_dir))
        img_path = os.path.join(save_dir, 'img_%d.jpg' % img_num)
        vutils.save_image(image, img_path, normalize=True)


    print('Saved img batch to %s' % save_dir)


def create_loader(data_dir, vocab, batch_size, tree_base_size, tree_branch_num):
    imsize = tree_base_size * (2 ** (tree_branch_num - 1))
    transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()
        ])

    dataloader = get_loader(root_dir=data_dir, method='val', vocab=vocab, batch_size=batch_size, transform=transform,
                            tree_base_size=tree_base_size, tree_branch_num=tree_branch_num)

    return dataloader


def gen_and_save_imgs(output_dir, batch_size, version, epoch, data_dir, models_dir, data_size):
    ##############
    # PARAMETERS #
    ##############
    gan_z_dim = 100
    text_embedding_dim = 256

    f = open(os.path.join(data_dir, 'big', 'vocab.pkl'), 'rb')
    vocab = pickle.load(f)

    f = open(os.path.join(data_dir, data_size, 'test/filenames.pickle'), 'rb')
    filenames = pickle.load(f)
    print("len filenames: ", len(filenames))

    # Get one caption per image
    captions = []
    cap_dir = os.path.join(data_dir, data_size, 'text')
    # Get corresponding caption
    for fname in filenames:
        with open('%s/%s.txt' % (cap_dir, fname)) as f:
            first_line = f.readline().rstrip()
            captions.append(first_line)

    cap_batches = [captions[x:x+batch_size] for x in range(0, len(captions), batch_size)]
    print(len(cap_batches))
    experimenter = Experimenter(z_dim=gan_z_dim, embedding_dim=text_embedding_dim,
                                models_dir=models_dir, version=version, data_dir=data_dir, vocab=vocab)
    for cap_batch in tqdm(cap_batches):
        images = experimenter.generate_images(epoch, sentences=cap_batch)[-1]
        save_images(output_dir, images)


if __name__ == '__main__':
    args = parse_args()
    gen_and_save_imgs(args.output_dir, args.batch_size, args.model_version, args.epoch, args.data_dir,
                      args.models_dir, args.data_size)
