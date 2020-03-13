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
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from transformers import AlbertTokenizer, AlbertModel

from processData import Vocabulary
from data_loader import get_loader


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
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    return parser.parse_args()


# loss
class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



#####################
# Encoder RASNET CNN
#####################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14,14))

    def forward(self, images):
        out = self.adaptive_pool(self.resnet(images))
        # batch_size, img size, imgs size, 2048
        out = out.permute(0, 2, 3, 1)
        return out


####################
# Attention Decoder
####################
class Decoder(nn.Module):
    def __init__(self, vocab_size, use_glove, use_albert):
        super(Decoder, self).__init__()
        self.encoder_dim = 2048
        self.attention_dim = 512
        self.use_albert = use_albert

        if use_glove:
            self.embed_dim = 300
        elif use_albert:
            self.embed_dim = 768

            # Load pretrained model tokenizer (vocabulary)
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

            # Load pre-trained model (weights)
            self.model = AlbertModel.from_pretrained('albert-base-v2').to(device)
            self.model.eval()
        else:
            self.embed_dim = 512

        self.decoder_dim = 512
        self.vocab_size = vocab_size
        self.dropout_rate = 0.5

        # soft attention
        self.enc_att = nn.Linear(2048, 512)
        self.dec_att = nn.Linear(512, 512)
        self.att = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # decoder layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        if not use_albert:
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            # load Glove embeddings
            if use_glove:
                glove_vectors = pickle.load(open(os.path.join(cfg.DATA_DIR, 'glove.6B/glove_words.pkl'), 'rb'))
                glove_vectors = torch.tensor(glove_vectors)

                self.embedding.weight = nn.Parameter(glove_vectors)

            # always fine-tune embeddings (even with GloVe)
            for p in self.embedding.parameters():
                p.requires_grad = True

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        dec_len = [x-1 for x in caption_lengths]
        max_dec_len = max(dec_len)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        embeddings = []
        # load albert or regular embeddings
        if not self.use_albert:
            embeddings = self.embedding(encoded_captions)
        elif self.use_albert:
            tokenizer = self.tokenizer
            model = self.model

            for cap_idx in encoded_captions:
                # padd caption to correct size
                while len(cap_idx) < max_dec_len:
                    cap_idx.append(cfg.VOCAB.PAD)

                cap = ' '.join([vocab.idx2word[word_idx.item()] for word_idx in cap_idx])
                cap = u'[CLS] '+cap



                tokenized_cap = tokenizer.tokenize(cap)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_cap)
                tokens_tensor = torch.tensor([indexed_tokens]).to(device)

                with torch.no_grad():
                    encoded_layers, _ = model(tokens_tensor)

                # TODO: Figure out why is wasn't "encoded_layers[11].squeeze(0)"
                # Maybe the model used previously had 12 layers, and so only the last layer was used..
                # Now, remove batch axis
                albert_embedding = encoded_layers.squeeze(0)

                split_cap = cap.split()
                tokens_embedding = []
                j = 0

                for full_token in split_cap:
                    curr_token = ''
                    x = 0
                    for i, _ in enumerate(tokenized_cap[1:]): # disregard CLS
                        token = tokenized_cap[i+j]
                        piece_embedding = albert_embedding[i+j]

                        # full token
                        if token.startswith('_') or token.startswith('<') or token.startswith('['):

                            if token.replace('_', '') == full_token and curr_token == '':
                                tokens_embedding.append(piece_embedding)
                                j += 1
                                break

                        else: # partial token
                            x += 1

                            if curr_token == '':
                                tokens_embedding.append(piece_embedding)
                                curr_token += token
                            else:
                                tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                                curr_token += token

                                if curr_token == full_token:
                                    j += x
                                    break

                cap_embedding = torch.stack(tokens_embedding)
                embeddings.append(cap_embedding)

            embeddings = torch.stack(embeddings)

        # init hidden state
        # init values described in page 4 Show-Attend-Tell paper
        avg_enc_out = encoder_out.mean(dim=1)
        h = self.h_lin(avg_enc_out)
        c = self.c_lin(avg_enc_out)

        predictions = torch.zeros(batch_size, max_dec_len, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_dec_len, num_pixels).to(device)

        for t in range(max(dec_len)):
            batch_size_t = sum([l > t for l in dec_len])

            # soft-attention
            enc_att = self.enc_att(encoder_out[:batch_size_t])
            dec_att = self.dec_att(h[:batch_size_t])
            att = self.att(self.relu(enc_att + dec_att.unsqueeze(1))).squeeze(2)
            alpha = self.softmax(att)
            attention_weighted_encoding = (encoder_out[:batch_size_t] * alpha.unsqueeze(2)).sum(dim=1)

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            batch_embeds = embeddings[:batch_size_t, t, :]
            cat_val = torch.cat([batch_embeds.double(), attention_weighted_encoding.double()], dim=1)

            h, c = self.decode_step(cat_val.float(), (h[:batch_size_t].float(), c[:batch_size_t].float()))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        # preds, sorted capts, dec lens, attention weights
        return predictions, encoded_captions, dec_len, alphas



###############
# Train model
###############

def train(encoder, decoder, decoder_optimizer, criterion, train_loader):
    print("Started training...")
    for epoch in tqdm(range(cfg.NUM_EPOCHS)):
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

            imgs = encoder(imgs.to(device))
            caps = caps.to(device)

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

            # save model each 100 batches
            if i % 5000 == 0 and i != 0:
                print('epoch ' + str(epoch + 1) + '/4 ,Batch ' + str(i) + '/' + str(num_batches) + ' loss:' + str(
                    losses.avg))

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


#################
# Validate model
#################

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

    if cfg.FROM_CHECKPOINT:
        encoder = Encoder().to(device)
        decoder = Decoder(vocab_size=len(vocabulary), use_glove=cfg.GLOVE_MODEL, use_albert=cfg.ALBERT_MODEL).to(device)

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
        encoder = Encoder().to(device)
        decoder = Decoder(vocab_size=len(vocabulary), use_glove=cfg.GLOVE_MODEL, use_albert=cfg.ALBERT_MODEL).to(device)
        decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=cfg.DECODER_LR)

    return encoder, decoder, decoder_optimizer


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

    # Activate the logger to have more information on what's happening under the hood
    import logging
    logging.basicConfig(level=logging.INFO)



    # Load vocabulary
    with open(os.path.join(cfg.DATA_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

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
        val_loader = get_loader('val', vocab, cfg.BATCH_SIZE)
        # Don't caluclate gradients for validation
        with torch.no_grad():
            validate(encoder=enc, decoder=dec, criterion=crit, val_loader=val_loader, root_dir=cfg.DATA_DIR)
