import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import os
import pprint
import time
import argparse
import dateutil.tz
from miscc.utils import mkdir_p, str2bool
from tqdm import tqdm
from datetime import datetime
from nltk.translate.bleu_score import corpus_bleu

from cfg.config import cfg, cfg_from_file
from model import Encoder, Decoder, CAPTION_CNN, CAPTION_RNN
from datasets import TextDataset, prepare_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file', default='cfg/pretrain_STREAM.yml',
                        help='optional config file', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='../data')
    parser.add_argument('--train', dest='train', type=str2bool)
    parser.add_argument('--val', dest='validate', type=str2bool)
    parser.add_argument('--preprocess_threshold', dest='threshold', type=int)
    return parser.parse_args()



def write_results(vocab, hypotheses, references, output_dir):
    dirname = os.path.join(output_dir, 'results')
    mkdir_p(dirname)
    number = len(os.listdir(dirname))
    filepath = '%s/%d.txt' % (dirname, number)
    file = open(filepath, 'w', encoding='utf8')
    print("hypotheses: ", hypotheses)
    print("references: ", references)
    references_len = [len(reference) for reference in references]
    hypotheses_len = [len(hyp) for hyp in hypotheses]
    print("len(references), references_len: ", len(references), ",", references_len)
    print("len(hypotheses), hypotheses_len: ", len(hypotheses), ",", hypotheses_len)
    for i in range(len(hypotheses)):
        hyp_sentence = []
        for word_idx in hypotheses[i]:
            print("word_idx: ", word_idx)
            hyp_sentence.append(vocab.idx2word[word_idx])

        ref_sentence = []
        for word_idx in references[i]:
            print("word_idx in references: ", word_idx)
            print("references[i]: ", references[i])
            ref_sentence.append(vocab.idx2word[word_idx])

        file.write("hypothesis: %s\n" % hyp_sentence)
        file.write("reference: %s\n\n" % ref_sentence)

    file.close()


def get_sentences(hypotheses, test_references, vocab):
    new_hypotheses = []
    new_references = []
    for i in range(len(hypotheses)):
        hyp = hypotheses[i]
        ref = test_references[i]

        hyp_sentence = []
        for word_idx in hyp:
            hyp_sentence.append(vocab.idx2word[word_idx])

        new_hypotheses.append(" ".join(hyp_sentence))

        ref_sentence = []
        for word_idx in ref:
            ref_sentence.append(vocab.idx2word[word_idx])
        new_references.append(" ".join(ref_sentence))

    return new_hypotheses, new_references

def calculate_bleu_scores(hypotheses, references):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    return bleu_1, bleu_2, bleu_3, bleu_4

def print_sample(hypotheses, references, test_references, imgs, alphas, k, show_att, losses, ixtoword):

    bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_scores(hypotheses, references)
    print(references[0][0])
    print("test_references: ", test_references)

    print("BLEU-1: " + str(bleu_1))
    print("BLEU-2: " + str(bleu_2))
    print("BLEU-3: " + str(bleu_3))
    print("BLEU-4: " + str(bleu_4))
    print("Validation loss: " + str(losses.avg))
    print("len hypotheses", len(hypotheses))

    cleaned_hypotheses = []
    cleaned_references = []
    for k in range(len(hypotheses)):
        hyp_sentence = []
        for word_idx in hypotheses[k]:
            hyp_sentence.append(ixtoword[word_idx])

        ref_sentence = []
        for word_idx in test_references[k]:
            ref_sentence.append(ixtoword[word_idx])

        hyp_sentence = " ".join(hyp_sentence)
        ref_sentence = " ".join(ref_sentence)

        cleaned_hypotheses.append(hyp_sentence)
        cleaned_references.append(ref_sentence)

    print("hypotheses: ", cleaned_hypotheses)
    print("references: ", cleaned_references)


    #hypotheses, test_references = get_sentences(hypotheses, test_references, vocab)
    return bleu_1, bleu_2, bleu_3, bleu_4

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

class bleu_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_loss_graph(epoch_num, losses, loss_dir):
    print("Epoch: ", epoch_num)
    x_values = range(1, len(losses) + 1)
    y_values = losses
    plt.plot(x_values, y_values)

    x_label = "Batch number in epoch %d" % epoch_num
    plt.xlabel(x_label)
    plt.ylabel("Losses")

    loss_path = os.path.join(loss_dir, 'epoch_%d' % epoch_num)
    plt.savefig(loss_path)


###############
# Train model
###############

def train(caption_cnn, caption_rnn, decoder_optimizer, criterion, train_loader, output_dir):
    loss_dir = os.path.join(output_dir, 'Losses')
    mkdir_p(loss_dir)
    print('output_dir: ', output_dir)

    print("Started training...")

    loss_list = []
    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):
        start_t = time.time()
        # Set the models in training mode
        caption_cnn.train()
        caption_rnn.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        # Loop through each batch
        for i, data in enumerate(tqdm(train_loader)):

            imgs, caps, cap_lens, class_ids, keys = prepare_data(data)
            # Extract imgs from list
            imgs = imgs[-1]

            # Skip last batch if it doesn't fit the batch size
            if len(imgs) != cfg.TRAIN.BATCH_SIZE:
                break

            encoder_out = caption_cnn(imgs.to(cfg.DEVICE))
            caps = caps.to(cfg.DEVICE)

            # Packing to optimize computations
            if cfg.CAP.USE_ORIGINAL:
                targets = pack_padded_sequence(caps, cap_lens, batch_first=True)[0]
                scores = caption_rnn(encoder_out, caps, cap_lens)  # 418 x 9956
            else:
                scores, caps_sorted, decode_lengths, alphas = caption_rnn(encoder_out, caps, cap_lens)
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                # Remove timesteps that we didn't decode at, or are pads
                targets = pack_padded_sequence(caps_sorted, decode_lengths, batch_first=True)[0]
            #scores, caps_sorted, decode_lengths, alphas = caption_rnn(encoder_out, caps, cap_lens)
            #scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            #targets = caps_sorted[:, 1:]
            #targets = pack_padded_sequence(caps_sorted, decode_lengths, batch_first=True)[0]
            #loss = criterion(scores, targets).to(cfg.DEVICE)

            loss = caption_loss(scores, targets).to(cfg.DEVICE) * cfg.TRAIN.SMOOTH.LAMBDA1
            if not cfg.CAP.USE_ORIGINAL:
                loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
            #losses.update(loss.item(), sum(cap_lens))

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-cfg.TRAIN.RNN_GRAD_CLIP, cfg.TRAIN.RNN_GRAD_CLIP)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(cap_lens))
            loss_list.append(loss.item())

            # save model each 5000 iterations
            if i % 5000 == 0 and i != 0:
                print('epoch ' + str(epoch + 1) + '/4 ,Batch ' + str(i) + '/' + str(num_batches) + ' loss:' + str(
                    losses.avg))

                # adjust learning rate (create condition for this)
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8

                print('saving model...')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': caption_rnn.state_dict(),
                    'optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(output_dir, 'decoder_mid'))

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': caption_cnn.state_dict(),
                    'loss': loss,
                }, os.path.join(output_dir, 'encoder_mid'))

                print('model saved')

        # save losses to graph
        save_loss_graph(epoch_num=epoch + 1, losses=loss_list, loss_dir=loss_dir)

        torch.save({
            'epoch': epoch,
            'model_state_dict': caption_rnn.state_dict(),
            'optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(output_dir, 'decoder_epoch%s' % str(epoch + 1)))

        torch.save({
            'epoch': epoch,
            'model_state_dict': caption_cnn.state_dict(),
            'loss': loss,
        }, os.path.join(output_dir, 'encoder_epoch%s' % str(epoch + 1)))

        end_t = time.time()
        print('epoch checkpoint saved')
        print('Time spent on epoch: %ds' % (end_t - start_t))

    print("Completed training...")


#################
# Validate model
#################

def validate(caption_cnn, caption_rnn, val_loader, ixtoword):
    print("Started validation...")
    caption_cnn.eval()
    caption_rnn.eval()

    losses = loss_obj()
    bleu_1 = bleu_obj()
    bleu_2 = bleu_obj()
    bleu_3 = bleu_obj()
    bleu_4 = bleu_obj()

    # Batches
    for i, data in enumerate(tqdm(val_loader)):

        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        references = []
        test_references = []
        hypotheses = []
        all_imgs = []
        all_alphas = []


        # Extract imgs from list
        imgs = imgs[-1]

        # Skip last batch if it doesn't fit the batch size
        if len(imgs) != cfg.TRAIN.BATCH_SIZE:
            break

        imgs_jpg = imgs.numpy()
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)

        # Forward prop.
        encoder_features = caption_cnn(imgs.to(cfg.DEVICE))
        captions = captions.to(cfg.DEVICE)


        if cfg.CAP.USE_ORIGINAL:
            print('Using original STREAM for validation')
            targets = captions
            targets_packed = pack_padded_sequence(captions, cap_lens, batch_first=True)[0]
            scores_packed = caption_rnn(encoder_features, captions, cap_lens) # 418 x 9956
            scores = scores_packed
        else:
            scores, caps_sorted, decode_lengths, alphas = caption_rnn(encoder_features, captions, cap_lens)
            targets = caps_sorted

            # Remove timesteps that we didn't decode at, or are pads
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        cap_loss = caption_loss(scores_packed, targets_packed) * cfg.TRAIN.SMOOTH.LAMBDA1
        if not cfg.CAP.USE_ORIGINAL:
            cap_loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
        losses.update(cap_loss.item(), sum(cap_lens))

        # References
        for j in range(targets.shape[0]):
            img_caps = targets[j].tolist()  # validation dataset only has 1 unique caption per img
            clean_cap = [w for w in img_caps if w != cfg.VOCAB.PAD]  # remove pad, start, and end
            img_captions = list(map(lambda c: clean_cap, img_caps))
            test_references.append(clean_cap)
            references.append(img_captions)

        # Hypotheses
        if cfg.CAP.USE_ORIGINAL:
            preds = caption_rnn.sample(encoder_features)
        else:
            _, preds = torch.max(scores, dim=2)
            if i == 0:
                all_alphas.append(alphas)
                all_imgs.append(imgs_jpg)

        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:cap_lens[j]]
            pred = [w for w in pred if w != cfg.VOCAB.PAD]
            if cfg.CAP.USE_ORIGINAL:
                pred = [w for w in pred if w != 96]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)


        score_1, score_2, score_3, score_4 = calculate_bleu_scores(hypotheses, references)
        bleu_1.update(score_1)
        bleu_2.update(score_2)
        bleu_3.update(score_3)
        bleu_4.update(score_4)

        if i % 10 == 0 or i == len(val_loader)-1:
            print("BLEU averages: ")
            print("BLEU-1: " + str(bleu_1.avg))
            print("BLEU-2: " + str(bleu_2.avg))
            print("BLEU-3: " + str(bleu_3.avg))
            print("BLEU-4: " + str(bleu_4.avg))
            print("Validation loss: " + str(losses.avg))
            print_sample(hypotheses, references, test_references, all_imgs, all_alphas, 1, False, losses, ixtoword)

    print("Completed validation...")


def caption_loss(cap_output, captions):
    criterion = nn.CrossEntropyLoss()
    caption_loss = criterion(cap_output, captions)
    return caption_loss

#############
# Init model
#############
def init_model(ixtoword):
    if cfg.CAP.USE_ORIGINAL:
        caption_cnn = CAPTION_CNN(embed_size=cfg.CAP.EMBED_SIZE)
        caption_rnn = CAPTION_RNN(embed_size=cfg.CAP.EMBED_SIZE, hidden_size=cfg.CAP.HIDDEN_SIZE,
                                  vocab_size=len(ixtoword), num_layers=cfg.CAP.NUM_LAYERS)
    else:
        caption_cnn = Encoder()
        caption_rnn = Decoder(idx2word=ixtoword)

    decoder_optimizer = torch.optim.Adam(params=caption_rnn.parameters(), lr=cfg.CAP.LEARNING_RATE)

    if cfg.CAP.CAPTION_CNN_PATH and cfg.CAP.CAPTION_RNN_PATH:
        print('Pre-Trained Caption Model')
        caption_cnn_checkpoint = torch.load(cfg.CAP.CAPTION_CNN_PATH, map_location=lambda storage, loc: storage)
        caption_rnn_checkpoint = torch.load(cfg.CAP.CAPTION_RNN_PATH, map_location=lambda storage, loc: storage)

        caption_cnn.load_state_dict(caption_cnn_checkpoint['model_state_dict'])
        caption_rnn.load_state_dict(caption_rnn_checkpoint['model_state_dict'])
        decoder_optimizer.load_state_dict(caption_rnn_checkpoint['optimizer_state_dict'])

    caption_cnn = caption_cnn.to(cfg.DEVICE)
    caption_rnn = caption_rnn.to(cfg.DEVICE)
    decoder_optimizer = decoder_optimizer

    return caption_cnn, caption_rnn, decoder_optimizer


def pretrain_STREAM():

    # rasnet tranformation/normalization
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip()
    ])
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(cfg.OUTPUT_PATH, cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    mkdir_p(output_dir)

    ######################
    # Run training/validation
    ######################
    crit = nn.CrossEntropyLoss().to(cfg.DEVICE)

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])


    split = 'train' if cfg.TRAIN.FLAG else 'test'
    dataset = TextDataset(cfg.DATA_DIR, split,
                      base_size=cfg.TREE.BASE_SIZE,
                      transform=image_transform)
    print(dataset.n_words, dataset.embeddings_num)
    assert dataset

    caption_cnn, caption_rnn, dec_optim = init_model(dataset.ixtoword)

    # Load data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    if cfg.TRAIN.FLAG:
        train(caption_cnn, caption_rnn, decoder_optimizer=dec_optim,
              criterion=crit, train_loader=dataloader, output_dir=output_dir)
    else:
        caption_cnn.eval()
        caption_rnn.eval()
        # Don't caluclate gradients for validation
        with torch.no_grad():
            validate(caption_cnn, caption_rnn, val_loader=dataloader, ixtoword=dataset.ixtoword)


def set_config_params(args):
    if args.cfg_file != '':
        cfg_from_file(args.cfg_file)

    if args.train is not None:
        cfg.TRAIN.FLAG = args.train

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    cfg.DATA_DIR = os.path.join(cfg.DATA_DIR, cfg.DATA_SIZE)

    # Set device
    if torch.cuda.is_available():
        cfg.DEVICE = torch.device('cuda')

    if args.threshold is not None:
        cfg.VOCAB.THRESHOLD = args.threshold

    print('Using config:')
    pprint.pprint(cfg)

if __name__ == '__main__':
    args = parse_args()
    set_config_params(args)


    pretrain_STREAM()

