'''
Parts of this code were incorporated from the following github repositories:
1. parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py
2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
3. ajamjooom/Image-Captions/data_loader.py
Link: https://github.com/ajamjoom/Image-Captions/blob/master/data_loader.py

This script has the Encoder and Decoder models and training/validation scripts.
Edit the parameters sections of this file to specify which models to load/run
'''

from cfg.config import cfg, cfg_from_file

import pickle
import argparse
import pprint
import os
import sys

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import torch
import torch.nn as nn
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from datetime import datetime

from processData import Vocabulary
from data_loader import get_loader
from model import Encoder, Decoder


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--root_dir', dest='root_dir', type=str, default='')
    # TODO: Use this
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    return parser.parse_args()


# loss
class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.losses = []

    def add_to_loss_list(self, loss):
        self.losses.append(loss)

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_losses(self):
        return self.losses


###############
# Train model
###############

def train(encoder, decoder, decoder_optimizer, criterion, train_loader):

    # TODO: Add scheduler? (See torch.optim how to adjust learning rate)
    print("Started training...")
    for epoch in tqdm(range(cfg.NUM_EPOCHS)):

        # Set the models in training mode
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        # Loop through each batch
        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

            imgs = encoder(imgs.to(device))
            caps = caps.to(device)

            # Packing to optimize computations
            scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets).to(device)

            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-cfg.GRAD_CLIP, cfg.GRAD_CLIP)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))
            losses.add_to_loss_list(loss.item())


            # TODO: Set this to 100?
            # save model each 100 batches
            if (i % 5000 == 0 and i != 0) or i == num_batches:
                print('epoch ' + str(epoch + 1) + '/4 ,Batch ' + str(i) + '/' + str(num_batches) + ' loss:' + str(
                    losses.avg))

                # save losses to graph
                save_loss_graph(epoch + 1, i + 1, losses.get_losses())

                # adjust learning rate (create condition for this)
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8

                print('saving model...')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': loss,
                }, './checkpoints/decoder_mid')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'loss': loss,
                }, './checkpoints/encoder_mid')

                print('model saved')

        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': loss,
        }, './checkpoints/decoder_epoch' + str(epoch + 1))

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'loss': loss,
        }, './checkpoints/encoder_epoch' + str(epoch + 1))

        print('epoch checkpoint saved')

    print("Completed training...")

def save_loss_graph(epoch_num, batch_num, losses):
    print("Epoch: ", epoch_num)
    x_values = range(1, len(losses)+1)
    y_values = losses
    plt.plot(x_values, y_values)

    x_label = "Batch number in epoch %d" % epoch_num
    plt.xlabel(x_label)
    plt.ylabel("Losses")
    date = datetime.now().strftime('%y-%m-%d')
    if not os.path.isdir('./checkpoints/losses/%s' % date):
        os.mkdir('./checkpoints/losses/%s' % date)
    plt.savefig('./checkpoints/losses/%s/epoch_%s_batch_%s' % (date, epoch_num, batch_num))


#################
# Validate model
#################

def write_results(hypotheses, references):
    dirname = './checkpoints/results/'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    number = len(os.listdir(dirname))
    date = str(datetime.now().strftime('%d-%m-%y'))
    filepath = './checkpoints/results/%d_%s.txt' % (number, date)
    file = open(filepath, 'w', encoding='utf8')
    for i in range(len(hypotheses)):
        hyp_sentence = []
        for word_idx in hypotheses[i]:
            hyp_sentence.append(vocab.idx2word[word_idx])

        ref_sentence = []
        for word_idx in references[i]:
            ref_sentence.append(vocab.idx2word[word_idx])

        file.write("hypothesis: %s\n" % hyp_sentence)
        file.write("reference: %s\n\n" % ref_sentence)

    file.close()


def print_sample(hypotheses, references, test_references, imgs, alphas, k, show_att, losses):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    print("Validation loss: " + str(losses.avg))
    print("BLEU-1: " + str(bleu_1))
    print("BLEU-2: " + str(bleu_2))
    print("BLEU-3: " + str(bleu_3))
    print("BLEU-4: " + str(bleu_4))

    write_results(hypotheses, test_references)

    '''
    img_dim = 336  # 14*24

    hyp_sentence = []
    for word_idx in hypotheses[k]:
        hyp_sentence.append(vocab.idx2word[word_idx])

    ref_sentence = []
    for word_idx in test_references[k]:
        ref_sentence.append(vocab.idx2word[word_idx])

    print('Hypotheses: ' + " ".join(hyp_sentence))
    print('References: ' + " ".join(ref_sentence))

    img = imgs[0][k]
    imageio.imwrite('img.jpg', img)
    if show_att:
        image = Image.open('img.jpg')
        image = image.resize([img_dim, img_dim], Image.LANCZOS)
        for t in range(len(hyp_sentence)):

            plt.subplot(np.ceil(len(hyp_sentence) / 5.), 5, t + 1)

            plt.text(0, 1, '%s' % (hyp_sentence[t]), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(image)
            current_alpha = alphas[0][t, :].detach().numpy()
            alpha = skimage.transform.resize(current_alpha, [img_dim, img_dim])
            if t == 0:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.7)
            plt.axis('off')
    else:
        img = imageio.imread('img.jpg')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    '''


def validate(encoder, decoder, criterion, val_loader):
    references = []
    test_references = []
    hypotheses = []
    all_imgs = []
    all_alphas = []


    print("Started validation...")
    decoder.eval()
    encoder.eval()

    losses = loss_obj()

    num_batches = len(val_loader)
    # Batches
    for i, (imgs, caps, caplens) in enumerate(tqdm(val_loader)):

        imgs_jpg = imgs.numpy()
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)

        # Forward prop.
        imgs = encoder(imgs.to(device))
        caps = caps.to(device)

        scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores_packed, targets_packed)
        loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
        losses.update(loss.item(), sum(decode_lengths))

        # References
        for j in range(targets.shape[0]):
            img_caps = targets[j].tolist()  # validation dataset only has 1 unique caption per img
            clean_cap = [w for w in img_caps if w not in [cfg.VOCAB.PAD, cfg.VOCAB.START, cfg.VOCAB.END]]  # remove pad, start, and end
            img_captions = list(map(lambda c: clean_cap, img_caps))
            test_references.append(clean_cap)
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decode_lengths[j]]
            pred = [w for w in pred if w not in [cfg.VOCAB.PAD, cfg.VOCAB.START, cfg.VOCAB.END]]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)

        if i == 0:
            all_alphas.append(alphas)
            all_imgs.append(imgs_jpg)

    print("Completed validation...")
    print_sample(hypotheses, references, test_references, all_imgs, all_alphas, 1, False, losses)


#############
# Init model
#############
def init_model(vocabulary):

    encoder = Encoder().to(device)
    decoder = Decoder(vocab=vocabulary, use_glove=cfg.GLOVE_MODEL, use_albert=cfg.ALBERT_MODEL).to(device)

    if cfg.FROM_CHECKPOINT:

        if torch.cuda.is_available():
            if cfg.ALBERT_MODEL:
                print('Pre-Trained ALBERT Model')
                encoder_checkpoint = torch.load('checkpoints/encoder_albert')
                decoder_checkpoint = torch.load('checkpoints/decoder_albert')
            elif cfg.GLOVE_MODEL:
                print('Pre-Trained GloVe Model')
                encoder_checkpoint = torch.load('checkpoints/encoder_glove')
                decoder_checkpoint = torch.load('checkpoints/decoder_glove')
            else:
                print('Pre-Trained Baseline Model')
                encoder_checkpoint = torch.load('checkpoints/encoder_baseline')
                decoder_checkpoint = torch.load('checkpoints/decoder_baseline')
        else:
            if cfg.ALBERT_MODEL:
                print('Pre-Trained ALBERT Model')
                encoder_checkpoint = torch.load('checkpoints/encoder_albert', map_location='cpu')
                decoder_checkpoint = torch.load('checkpoints/decoder_albert', map_location='cpu')
            elif cfg.GLOVE_MODEL:
                print('Pre-Trained GloVe Model')
                encoder_checkpoint = torch.load('checkpoints/encoder_glove', map_location='cpu')
                decoder_checkpoint = torch.load('checkpoints/decoder_glove', map_location='cpu')
            else:
                print('Pre-Trained Baseline Model')
                encoder_checkpoint = torch.load('checkpoints/encoder_fbaseline', map_location='cpu')
                decoder_checkpoint = torch.load('checkpoints/decoder_baseline', map_location='cpu')

        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=cfg.DECODER_LR)
        decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
        decoder_optimizer.load_state_dict(decoder_checkpoint['optimizer_state_dict'])

    else:
        decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=cfg.DECODER_LR)

    return encoder, decoder, decoder_optimizer

def main():
    crit = nn.CrossEntropyLoss().to(device)
    enc, dec, dec_optim = init_model(vocab)


    ######################
    # Run training/validation
    ######################

    if cfg.TRAIN_MODEL:
        # Load data
        train_loader = get_loader('train', vocab, cfg.BATCH_SIZE, root_dir=cfg.DATA_DIR)
        train(encoder=enc, decoder=dec, decoder_optimizer=dec_optim,
              criterion=crit, train_loader=train_loader)

    if cfg.VALID_MODEL:
        # Load data
        val_loader = get_loader('val', vocab, cfg.BATCH_SIZE, root_dir=cfg.DATA_DIR)
        # Don't caluclate gradients for validation
        with torch.no_grad():
            validate(encoder=enc, decoder=dec, criterion=crit, val_loader=val_loader)


if __name__ == '__main__':
    args = parse_args()
    # Get file and set config
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.root_dir != '':
        cfg.ROOT_DIR = args.ROOT_DIR
    if args.data_size != '':
        cfg.DATASET_SIZE = args.data_size

    cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, cfg.DATASET_SIZE)
    print('Using config:')
    pprint.pprint(cfg)

    # Load vocabulary
    with open(os.path.join(cfg.DATA_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    main()





