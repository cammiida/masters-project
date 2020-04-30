import os
import torch
from tqdm import tqdm
from datetime import datetime
import dateutil.tz

from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
import numpy as np

from cfg.config import cfg
from STREAM.data_processer import print_sample
from miscc.utils import mkdir_p

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


def save_loss_graph(epoch_num, losses, output_dir):
    print("Epoch: ", epoch_num)
    x_values = range(1, len(losses) + 1)
    y_values = losses
    plt.plot(x_values, y_values)

    x_label = "Batch number in epoch %d" % epoch_num
    plt.xlabel(x_label)
    plt.ylabel("Losses")
    loss_dir = os.path.join(output_dir, 'losses')
    if not os.path.isdir(loss_dir):
        mkdir_p(loss_dir)

    loss_path = os.path.join(loss_dir, 'epoch', str(epoch_num))
    plt.savefig(loss_path)

###############
# Train model
###############

def train(encoder, decoder, decoder_optimizer, criterion, train_loader):
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/output/%s/%s_%s_%s' % \
                 (cfg.OUTPUT_PATH, cfg.DATASET_SIZE,
                  cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    print('output_dir: ', output_dir)

    # TODO: Add scheduler? (See torch.optim how to adjust learning rate)
    print("Started training...")

    loss_list = []
    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):

        # Set the models in training mode
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        # Loop through each batch
        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):
            # Extract imgs from list
            imgs = imgs[-1]
            imgs = encoder(imgs.to(cfg.DEVICE))
            caps = caps.to(cfg.DEVICE)

            # Skip last batch if it doesn't fit the batch size
            if len(imgs) != cfg.TRAIN.BATCH_SIZE:
                break

            # Packing to optimize computations
            scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets).to(cfg.DEVICE)

            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-cfg.STREAM.GRAD_CLIP, cfg.STREAM.GRAD_CLIP)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))
            loss_list.append(loss.item())


            # TODO: Set this to 100?
            # save model each 100 batches
            if (i % 5000 == 0 and i != 0) or i == num_batches:
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
                }, os.path.join(output_dir, 'decoder_mid'))

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'loss': loss,
                }, os.path.join(output_dir, 'encoder_mid'))

                print('model saved')

        # save losses to graph
        save_loss_graph(epoch_num=epoch + 1, losses=loss_list, output_dir=output_dir)

        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(output_dir, 'decoder_epoch', str(epoch + 1)))

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'loss': loss,
        }, os.path.join(output_dir, 'encoder_epoch', str(epoch + 1)))

        print('epoch checkpoint saved')

    print("Completed training...")


#################
# Validate model
#################

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
        imgs = encoder(imgs.to(cfg.DEVICE))
        caps = caps.to(cfg.DEVICE)

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
