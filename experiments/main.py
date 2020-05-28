import datetime
import math
import os

import dateutil.tz
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from experiments.experimenter import Experimenter
from miscc.utils import mkdir_p


def gen_out_path(output_dir: str, version: str, sent_type: str, epoch: int, title: str):
    title = title.lower().replace(' ', '_').replace('.', '')
    output_dir = os.path.join(output_dir, version, sent_type, title)
    mkdir_p(output_dir)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d')
    num_files = 0

    for f_name in os.listdir(output_dir):
        if 'epoch_%d' % epoch in f_name:
            num_files += 1

    print('num_files: ', num_files)

    file_name = 'epoch_%d_%s_%d' % (epoch, timestamp, num_files)
    output_path = os.path.join(output_dir, file_name)
    print('output_path: ', output_path)
    return output_path


def gen_pil_imgs(imgs: torch.Tensor) -> list:
    """
    :param imgs: torch tensor with size: batch x channels x height x width
    :return: list of PIL Images
    """
    pil_imgs = []
    for img in imgs:
        im = img.data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        pil_imgs.append(im)

    return pil_imgs


def save_img_grid(imgs: torch.Tensor, epoch: int, version: str, sent_type: str, output_dir: str, title='Generated Images'):
    """
        :param imgs: PyTorch tensor with stacked images of same size
        :param experiment: experiment number
        :param version: original or new generator
        :param title: Title for the plot
        :return: None
    """
    output_path = gen_out_path(output_dir, version, sent_type, epoch, title)

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

def mul_img_one_sent(experimenter: Experimenter, sentence: str, epoch: int, num_imgs: int, output_dir: str, version: str, sent_type: str):
    imgs = torch.zeros(num_imgs, 3, 256, 256)
    for i in range(num_imgs):
        imgs[i] = experimenter.generate_images([sentence], epoch=epoch)[-1]

    save_img_grid(imgs=imgs, output_dir=output_dir, epoch=epoch, version=version, sent_type=sent_type, title=sentence)
    return imgs


def mul_imgs_mul_sent(experimenter: Experimenter, sentences: dict, epoch: int, num_imgs: int, output_dir: str, version, sent_type: str):
    # Generate num_img per sentence. Save each separately?

    img_dict = {}
    # Generate the four images per sentence
    for sent in sentences:
        sent_imgs = mul_img_one_sent(experimenter, sent, epoch, num_imgs=4, output_dir=output_dir, version=version, sent_type=sent_type)
        img_dict[sent] = sent_imgs

    return img_dict


def exp(experimenter: Experimenter, sentences: dict, start_epoch: int, max_epoch: int, epoch_inc: int,
        version: str, sent_type: str, output_dir: str, num_imgs: int = 4):
    epochs = [i * epoch_inc for i in range((max_epoch - start_epoch)//epoch_inc + 1)]
    sentences = sentences[sent_type]
    img_dict = {}
    for epoch in tqdm(epochs):
        img_dict[epoch]= mul_imgs_mul_sent(experimenter, sentences, epoch, num_imgs, version=version, sent_type=sent_type, output_dir=output_dir)




# Parameters
sentence_category = 'descriptive'
start_epoch = 0
max_epoch = 34
epoch_inc = 10
num_imgs = 4

data_dir = '../data'
models_dir = '../models'
output_dir = '../output/experiments'
data_size = 'big'

gan_z_dim = 100
text_embedding_dim = 256

experimenter_new = Experimenter(z_dim=gan_z_dim, embedding_dim=text_embedding_dim,
                                models_dir=models_dir,
                                version='new', data_dir=data_dir, data_size=data_size, epoch=max_epoch)

experimenter_original = Experimenter(z_dim=gan_z_dim, embedding_dim=text_embedding_dim,
                                     models_dir=models_dir,
                                     version='original', data_dir=data_dir, data_size=data_size,
                                     epoch=max_epoch)

sentences = {
    'descriptive': [
        'A bunch of vehicles that are in the street.',
        'A street that goes on to a high way with the light on red.',
        'A large white teddy bear sitting on top of an SUV.',
        'A stationary train with the door wide open.',
        #'A train on some tracks with power lines above it.',
        #'An old truck carrying luggage at the back',
        'A lamp with a shade sitting on top of an older model television.',
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
exp(experimenter_new, sentences, start_epoch=start_epoch, max_epoch=max_epoch,
    epoch_inc=epoch_inc, version='new', sent_type='descriptive', output_dir=output_dir, num_imgs=num_imgs)

#exp(experimenter_original, sentences, start_epoch=start_epoch, max_epoch=max_epoch,
#    epoch_inc=epoch_inc, version='original', sent_type='descriptive', output_dir=output_dir, num_imgs=num_imgs)


# NONDESCRIPTIVE SENTENCES
exp(experimenter_new, sentences, start_epoch=start_epoch, max_epoch=max_epoch,
    epoch_inc=epoch_inc, version='new', sent_type='nondescriptive', output_dir=output_dir, num_imgs=num_imgs)

#exp(experimenter_original, sentences, start_epoch=start_epoch, max_epoch=max_epoch,
#    epoch_inc=epoch_inc, version='original', sent_type='nondescriptive', output_dir=output_dir, num_imgs=num_imgs)




