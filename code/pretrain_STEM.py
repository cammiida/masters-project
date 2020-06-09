from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from cfg.config import cfg, cfg_from_file

from process_data import Vocabulary
from datasets import get_loader

from model import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import pickle
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a MirrorGAN network')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='cfg/pretrain_STEM.yml', type=str)
    parser.add_argument('--root_data_dir', dest='root_data_dir', type=str, default='')
    parser.add_argument('--data_size', dest='data_size', type=str, default='')
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    return parser.parse_args()


def build_models():
    text_encoder = RNN_ENCODER(len(vocab), nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = torch.LongTensor(range(cfg.TRAIN.BATCH_SIZE))

    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)

    text_encoder = text_encoder.to(cfg.DEVICE)
    image_encoder = image_encoder.to(cfg.DEVICE)
    labels = labels.to(cfg.DEVICE)

    return text_encoder, image_encoder, labels, start_epoch


def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, image_dir):
    print("Training...")

    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    w_losses = []
    s_losses = []
    count = (epoch + 1) * len(dataloader)

    for step, data in enumerate(tqdm(dataloader)):
        start_time = time.time()
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens = data
        # Don't use the last batch if it is smaller than batch_size
        if captions.shape[0] != batch_size:
            break

        # Extract imgs from list (get highest res images)
        imgs = imgs[-1]
        # --> batch_size x channels x width x height
        imgs = imgs.to(cfg.DEVICE)
        captions = captions.to(cfg.DEVICE)

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs)
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, cap_lens,
                                                 class_ids=None, batch_size=cfg.TRAIN.BATCH_SIZE)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        w_loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids=None, batch_size=cfg.TRAIN.BATCH_SIZE)
        s_loss = s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data

        loss = w_loss + s_loss
        w_losses.append(w_loss.item())
        s_losses.append(s_loss.item())
        #
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            # attention Maps
            img_set, _ = \
                build_super_images(imgs.cpu(), captions, vocab.idx2word, attn_maps, att_sze)
                #build_super_images(imgs[-1].cpu(), captions,
                #                   vocab.idx2word, attn_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                im.save(fullpath)

    return count, w_losses, s_losses


def evaluate(dataloader, cnn_model, rnn_model, batch_size, labels):
    print("Evaluating...")
    cnn_model.eval()
    rnn_model.eval()

    s_total_loss = 0
    w_total_loss = 0

    for step, data in enumerate(tqdm(dataloader)):
        real_imgs, captions, cap_lens = data

        # Don't use the last batch if it is smaller than batch_size
        if captions.shape[0] != batch_size:
            break

        # Extract imgs from list
        real_imgs = real_imgs[-1]
        real_imgs = real_imgs.to(cfg.DEVICE)
        captions = captions.to(cfg.DEVICE)

        words_features, sent_code = cnn_model(real_imgs)
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        # TODO: Check if class_ids should be provided
        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids=None, batch_size=batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        # TODO: Check if class_ids should be provided
        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids=None, batch_size=batch_size)
        s_total_loss += (s_loss0 + s_loss1).data


        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss

def save_losses(losses: dict, epoch, save_dir):
    print("Saving losses...")
    plt.figure(figsize=(10, 5))
    plt.title("STEM Word and Sentence Loss During Training")
    for loss in losses:
        plt.plot(losses[loss], label=loss)
        plt.plot(losses[loss], label=loss)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    save_dir = os.path.join(save_dir, 'Losses')
    if not os.path.isdir(save_dir):
        mkdir_p(save_dir)

    losses_name = 'STEM_losses_epoch_%d' % epoch
    losses_path = os.path.join(save_dir, losses_name)
    plt.savefig(losses_path)


def main():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/output/%s/%s_%s_%s' % \
                 (cfg.OUTPUT_PATH, cfg.DATASET_SIZE,
                  cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    print('output_dir: ', output_dir)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)
    w_losses = []
    s_losses = []

    batch_size = cfg.TRAIN.BATCH_SIZE

    # Get data loaders ###################################################
    train_loader = get_loader(cfg.DATA_DIR, 'train', vocab, batch_size, transform, tree_base_size=cfg.TREE.BASE_SIZE,
                              tree_branch_num=cfg.TREE.BRANCH_NUM)
    val_loader = get_loader(cfg.DATA_DIR, 'val', vocab, batch_size, transform, tree_base_size=cfg.TREE.BASE_SIZE,
                            tree_branch_num=cfg.TREE.BRANCH_NUM)

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    # At any point you can hit Ctrl + C to break out of training early
    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            start_time = time.time()
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            # epoch_start_time = time.time()
            count, batch_w_losses, batch_s_losses = \
                train(train_loader, image_encoder, text_encoder, batch_size, labels, optimizer, epoch, image_dir)

            w_losses = w_losses + batch_w_losses
            s_losses = s_losses + batch_s_losses


            print('-' * 89)
            if len(val_loader) > 0:
                s_loss, w_loss = \
                    evaluate(val_loader, image_encoder, text_encoder, batch_size, labels)

                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')

                losses = {'w_losses': w_losses, 's_losses': s_losses}
                save_losses(losses, epoch, output_dir)
            end_time = time.time()
            print("Time spent on training for one epoch: %ds" % (end_time-start_time))
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == '__main__':
    # Get file
    args = parse_args()
    if args.cfg_file != '':
        cfg_from_file(args.cfg_file)

    # Set config
    if args.root_data_dir != '':
        cfg.ROOT_DATA_DIR = args.root_data_dir
    if args.data_size != '':
        cfg.DATASET_SIZE = args.data_size

    cfg.DATA_DIR = os.path.join(cfg.ROOT_DATA_DIR, cfg.DATASET_SIZE)

    print('Using config:')
    pprint.pprint(cfg)

    # Set seed for train or not train env
    if not cfg.TRAIN.FLAG:
        args.manual_seed = 100
    elif args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    # Set device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manual_seed)
        cfg.DEVICE = torch.device('cuda')
    else:
        cfg.DEVICE = torch.device('cpu')
    cudnn.benchmark = True

    # Load vocabulary
    with open(os.path.join(cfg.DATA_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    # rasnet transformation/normalization
    transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])

    main()















