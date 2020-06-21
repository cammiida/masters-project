import argparse
import datetime
import os
import pprint

import dateutil.tz
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import torchvision.transforms as transforms

from cfg.config import cfg, cfg_from_file
from experiments.experimenter import Experimenter
from miscc.utils import mkdir_p
from datasets import TextDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                        default='../cfg/experiments/gen_imgs_new.yml', type=str)
    args = parser.parse_args()
    return args


def gen_out_path(sent_type: str, epoch: int, title: str):
    title = title.lower().replace(' ', '_').replace('.', '')
    output_dir = os.path.join(cfg.OUTPUT_PATH, 'gen_imgs', cfg.CONFIG_NAME, sent_type, title)
    mkdir_p(output_dir)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d')
    num_files = 0

    for f_name in os.listdir(output_dir):
        if 'epoch_%d' % epoch in f_name:
            num_files += 1

    file_name = 'epoch_%d_%s_%d' % (epoch, timestamp, num_files)
    output_path = os.path.join(output_dir, file_name)
    print('output_path: ', output_path)
    return output_path


def save_img_grid(imgs: torch.Tensor, epoch: int, sent_type: str, title='Generated Images'):
    """
        :param imgs: PyTorch tensor with stacked images of same size
        :param experiment: experiment number
        :param version: original or new generator
        :param title: Title for the plot
        :return: None
    """
    output_path = gen_out_path(sent_type, epoch, title)

    img_grid = vutils.make_grid(imgs, padding=2, normalize=True).cpu()
    #plt.figure(figsize=(16, 16))
    #fig, axes = plt.subplots(1, 4, figsize=(9,3))
    plt.axis('off')
    plt.title('Epoch %d' % epoch)
    plt.imshow(img_grid.permute(1, 2, 0))

    plt.savefig(output_path, bbox_inches='tight',pad_inches = 0,)



def display_img_grid(imgs: torch.Tensor, title='Generated Images'):
    """
    :param imgs: PyTorch tensor with stacked images of same size
    :param title: Title for the plot
    :return: None
    """
    # Print images
    img_grid = vutils.make_grid(imgs, padding=2, normalize=True).cpu()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.subplot()
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))

    plt.show()


###############
# EXPERIMENTS #
###############

def mul_img_one_sent(experimenter: Experimenter, sentence: str, g_path: str, epoch: int, num_imgs: int, sent_type: str):
    imgs = torch.zeros(num_imgs, 3, 256, 256)
    for i in range(num_imgs):
        imgs[i] = experimenter.generate_images(g_path, sentences=[sentence], z_dim=cfg.GAN.Z_DIM)[-1]

    #i_score = calculate_inception_score(imgs)
    save_img_grid(imgs=imgs, epoch=epoch, sent_type=sent_type, title=sentence)
    return imgs


def mul_imgs_mul_sent(experimenter: Experimenter, sentences: dict, g_dir: str, epoch: int, num_imgs: int, sent_type: str):
    # Generate num_img per sentence. Save each separately?
    g_path = os.path.join(g_dir, 'netG_epoch_%d.pth' % epoch)
    img_dict = {}
    # Generate the four images per sentence
    for sent in sentences:
        sent_imgs = mul_img_one_sent(experimenter, sent, g_path, epoch, num_imgs=num_imgs, sent_type=sent_type)
        img_dict[sent] = sent_imgs

    return img_dict


def exp(experimenter: Experimenter, sentences: dict, g_dir: str, start_epoch: int, max_epoch: int, epoch_inc: int,
        sent_type: str, num_imgs: int = 4):
    epochs = [(i * epoch_inc) + start_epoch for i in range((max_epoch - start_epoch)//epoch_inc + 1)]
    sentences = sentences[sent_type]
    img_dict = {}
    for epoch in tqdm(epochs):
        img_dict[epoch]= mul_imgs_mul_sent(experimenter, sentences, g_dir, epoch, num_imgs, sent_type=sent_type)



if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    # Parameters
    # TODO: Add this to and get from cfg
    sentence_category = 'descriptive'
    start_epoch = 0
    max_epoch = 160
    epoch_inc = 32
    num_imgs = 4



    g_dir = cfg.TRAIN.NET_G

    data_dir = os.path.join(cfg.DATA_DIR, cfg.DATA_SIZE)

    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    data_dir = os.path.join(cfg.DATA_DIR, cfg.DATA_SIZE)

    dataset = TextDataset(data_dir, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    experimenter_new = Experimenter(embedding_dim=cfg.TEXT.EMBEDDING_DIM, net_E=cfg.TRAIN.NET_E,
                                    n_words=dataset.n_words, wordtoix=dataset.wordtoix)

    experimenter_original = Experimenter(embedding_dim=cfg.TEXT.EMBEDDING_DIM, net_E=cfg.TRAIN.NET_E,
                                         n_words=dataset.n_words, wordtoix=dataset.wordtoix)

    sentences = {
        'descriptive': [
            'a skier with a red jacket on going down the side of a mountain',
            'the pizza is cheesy with pepperoni for the topping',
            'boats at the dock with a city backdrop',
            'brown horses are running on a green field',
            'A bunch of vehicles that are in the street.',
            'A street that goes on to a high way with the light on red.',
            'A large white teddy bear sitting on top of an SUV.',
            'A stationary train with the door wide open.',
            'A lamp with a shade sitting on top of an older model television.',

                #'A train on some tracks with power lines above it.',
                #'An old truck carrying luggage at the back',
                #'A pot filled with some liquid and parchment paper.',
                #'A hand holding a smart phone with apps on a screen.'
        ],
        'nondescriptive': [
            'It is utterly terrific that he got accepted to that school.',
            'The weather has been really horrible these last few days.',
            'Today we are going to do some shopping for Lucy\'s prom.',
            'We are about five minutes late, I\'m afraid.',
            'I love going to art galleries and seeing all the beautiful and strange creations they have there.'
        ]
    }

    # DESCRIPTIVE SENTENCES
    #exp(experimenter_new, sentences, start_epoch=start_epoch, g_dir=g_dir, max_epoch=max_epoch,
    #    epoch_inc=epoch_inc, sent_type='descriptive', num_imgs=num_imgs)

    #exp(experimenter_original, sentences, g_dir=g_dir, start_epoch=start_epoch, max_epoch=max_epoch,
    #    epoch_inc=epoch_inc, sent_type='descriptive', num_imgs=num_imgs)


    # NONDESCRIPTIVE SENTENCES
    exp(experimenter_new, sentences, g_dir=g_dir, start_epoch=start_epoch, max_epoch=max_epoch,
        epoch_inc=epoch_inc, sent_type='nondescriptive', num_imgs=num_imgs)

    #exp(experimenter_original, sentences, g_dir=g_dir, start_epoch=start_epoch, max_epoch=max_epoch,
    #    epoch_inc=epoch_inc, version='original', sent_type='nondescriptive', num_imgs=num_imgs)




