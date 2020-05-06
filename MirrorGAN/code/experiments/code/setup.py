import torch
import os
import pickle
import nltk

from cfg.config import cfg
from datasets import Vocabulary

from model import G_NET, RNN_ENCODER
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import torchvision.utils as vutils


class Experimenter():
    def __init__(self, model_path, data_path, data_size):
        self.nz = cfg.GAN.Z_DIM

        # Load vocabulary
        vocab_path = os.path.join(data_path, data_size, 'vocab.pkl')
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Load models
        self.text_encoder, self.netG = self.build_models(model_path, data_size)


    def build_models(self, model_path, data_size):
        # Load trained text encoder model
        text_encoder = RNN_ENCODER(len(self.vocab), nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(model_path, data_size, 'STEM/text_encoder.pth'),
                                map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        text_encoder.eval()

        # Load trained generator model
        netG = G_NET()
        state_dict = torch.load(os.path.join(model_path, data_size, 'MirrorGAN/netG_epoch_10.pth'),
                                map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        netG.eval()

        return text_encoder, netG

    def sent_to_target_ids(self, sentences: list) -> (torch.Tensor, list):
        # Prepare sentences
        # TODO: Change this to BERT?
        tokenized_sentences = []
        for sent in sentences:
            tokens = nltk.tokenize.word_tokenize(str(sent).lower())
            caption = list()
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            target = torch.Tensor(caption)
            tokenized_sentences.append(target)

        # Sort longest to shortest sentence
        tokenized_sentences.sort(reverse=True, key=len)

        lengths = [len(sent) for sent in tokenized_sentences]

        targets = torch.zeros(len(tokenized_sentences), max(lengths)).long()
        for i, cap in enumerate(tokenized_sentences):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return targets, lengths



    def generate_images(self, sentences):
        targets, target_lengths = self.sent_to_target_ids(sentences)
        # reset text_encoder
        batch_size = len(sentences)
        hidden = self.text_encoder.init_hidden(batch_size)

        # words_embs: batch_size x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = self.text_encoder(targets, target_lengths, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        noise = torch.FloatTensor(batch_size, self.nz)
        mask = (targets == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        # (2) Generate fake_images
        noise.data.normal_(0, 1)
        with torch.no_grad():
            fake_imgs, attn_maps, _, _ = self.netG(noise, sent_emb, words_embs, mask)


        return fake_imgs


    def gen_pil_imgs(self, imgs: torch.Tensor) -> list:
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
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.show()

def main():
    model_path = '../../../../models/'
    data_path = '../../../../data/'
    data_size = 'big'


    sentences = [
        'A bunch of vehicles that are in the street.',
        'A street that goes on to a high way with the light on red.',
        'A large white teddy bear sitting on top of an SUV.',
        'A stationary train with the door wide open.',
        'A train on some tracks with power lines above it.',
        'An old truck carrying luggage at the back',
        'A lamp with a shade sitting on top of an older model television.',
        'A pot filled with some liquid and parchment paper.',
        'A hand holding a smart phone with apps on a screen.'
    ]

    #sentences = ['A red cabin on an island with a single boat by the pier.']

    experimenter = Experimenter(model_path, data_path, data_size)
    '''
    fake_imgs = experimenter.generate_images(sentences)

    # Get highest resolution images
    fake_imgs = fake_imgs[-1]
    display_img_grid(fake_imgs)
    '''
    experiment1(sentences, experimenter)

# Generate many images of same sentence
def experiment1(sentences, experimenter, num_imgs=6):

    for sent in sentences:
        fake_imgs = torch.zeros(num_imgs, 3, 256, 256)
        for i in range(num_imgs):
            fake_img = experimenter.generate_images([sent])
            fake_img = fake_img[-1].squeeze()
            fake_imgs[i] = fake_img
        display_img_grid(fake_imgs, title=sent)


if __name__ == '__main__':
    main()




