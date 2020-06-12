from __future__ import print_function
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt
from cfg.config import cfg
from model import G_NET, D_NET64, D_NET128, D_NET256, RNN_ENCODER, CNN_ENCODER, CAPTION_CNN, CAPTION_RNN
from model import Encoder, Decoder
from model import MyDataParallel
from miscc.utils import mkdir_p, weights_init, load_params, copy_G_params
from miscc.utils import build_super_images, build_super_images2
from miscc.losses import words_loss, discriminator_loss, generator_loss, KL_loss
from process_data import Vocabulary
import os
import time
import numpy as np
from tqdm import tqdm


# MirrorGAN
class Trainer(object):
    def __init__(self, output_dir, data_loader, vocab):
        if cfg.TRAIN.FLAG:
            self.output_dir = output_dir
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.losses_dir = os.path.join(output_dir, 'Losses')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.losses_dir)

        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.vocab = vocab
        self.n_words = len(vocab)
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.ixtoword = vocab.idx2word

    def build_models(self):
        #####################
        ##  TEXT ENCODERS  ##
        #####################

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from: ', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from: ', cfg.TRAIN.NET_E)
        text_encoder.eval()

        ######################
        ##  CAPTION MODELS  ##
        ######################

        # cnn_encoder and rnn_encoder
        if cfg.TRAIN.STREAM.USE_ORIGINAL:
            caption_cnn = CAPTION_CNN(embed_size=cfg.TEXT.EMBEDDING_DIM)
            caption_rnn = CAPTION_RNN(embed_size=cfg.TEXT.EMBEDDING_DIM, hidden_size=cfg.TRAIN.STREAM.HIDDEN_SIZE,
                                      vocab_size=len(self.vocab), num_layers=cfg.TRAIN.STREAM.NUM_LAYERS)


            caption_cnn_checkpoint = torch.load(cfg.TRAIN.CAP_CNN, map_location=lambda storage, loc: storage)
            caption_cnn.load_state_dict(caption_cnn_checkpoint)
            caption_rnn_checkpoint = torch.load(cfg.TRAIN.CAP_RNN, map_location=lambda storage, loc: storage)
            caption_rnn.load_state_dict(caption_rnn_checkpoint)
        else:
            caption_cnn = Encoder()
            caption_cnn_checkpoint = torch.load(cfg.TRAIN.CAP_CNN, map_location=lambda storage, loc: storage)
            caption_cnn.load_state_dict(caption_cnn_checkpoint['model_state_dict'])

            caption_rnn = Decoder(vocab=self.vocab)
            caption_rnn_checkpoint = torch.load(cfg.TRAIN.CAP_RNN, map_location=lambda storage, loc: storage)
            caption_rnn.load_state_dict(caption_rnn_checkpoint['model_state_dict'])

        for p in caption_cnn.parameters():
            p.requires_grad = False
        print('Load caption model from: ', cfg.TRAIN.CAP_CNN)
        caption_cnn.eval()

        #caption_rnn = CAPTION_RNN(cfg.TEXT.EMBEDDING_DIM, cfg.TRAIN.STREAM.HIDDEN_SIZE * 2, self.N_WORDS, cfg.TREE.BRANCH_NUM)
        for p in caption_rnn.parameters():
            p.requires_grad = False
        print('Load caption model from: ', cfg.TRAIN.CAP_RNN)

        ###################################
        ##  GENERATOR AND DISCRIMINATOR  ##
        ###################################
        netsD = []
        # TODO: Add the BDCGAN thing?
        netG = G_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(D_NET64())
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(D_NET128())
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(D_NET256())

        netG.apply(weights_init)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        print('# of netsD', len(netsD))

        epoch = 0
        # Load models from checkpoints
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)


        text_encoder = text_encoder.to(cfg.DEVICE)
        image_encoder = image_encoder.to(cfg.DEVICE)
        caption_cnn = caption_cnn.to(cfg.DEVICE)
        caption_rnn = caption_rnn.to(cfg.DEVICE)
        netG.to(cfg.DEVICE)
        for i in range(len(netsD)):
            netsD[i].to(cfg.DEVICE)

        return [text_encoder, image_encoder, caption_cnn, caption_rnn, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD


    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = torch.FloatTensor(batch_size).fill_(1)
        fake_labels = torch.FloatTensor(batch_size).fill_(0)
        match_labels = torch.LongTensor(range(batch_size))

        real_labels = real_labels.to(cfg.DEVICE)
        fake_labels = fake_labels.to(cfg.DEVICE)
        match_labels = match_labels.to(cfg.DEVICE)

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.module.state_dict(),
                   '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)

        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.module.state_dict(),
                       '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')


    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires


    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.vocab.idx2word,
                                   attn_maps, att_sze, lr_imgs=lr_img)

            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png' \
                            % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)

        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.vocab.idx2word, att_maps, att_sze)

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png' \
                        % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        print("Started training with %d GPUS" % torch.cuda.device_count())
        text_encoder, image_encoder, caption_cnn, caption_rnn, netG, netsD, start_epoch = \
            self.build_models()

        # Parallelize netG and netsD models
        netG = MyDataParallel(netG)
        for i in range(len(netsD)):
            netsD[i] = MyDataParallel(netsD[i])

        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = torch.FloatTensor(batch_size, nz)
        fixed_noise = torch.FloatTensor(batch_size, nz).normal_(0, 1)

        noise, fixed_noise = noise.to(cfg.DEVICE), fixed_noise.to(cfg.DEVICE)

        G_losses = []
        D_losses = []
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for batch_num, data in enumerate(tqdm(self.data_loader)):
                # Skip last batch in case batch size doesn't divide length of data
                if batch_num == len(self.data_loader) - 1:
                    break

                # (1) Prepare training data and compute text embeddings
                imgs, captions, cap_lens = data
                captions = captions.to(cfg.DEVICE)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                # (2) Generate fake_images
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                # (3) Update D network
                errD_total = 0
                D_logs = ''
                for j in range(len(netsD)):
                    netsD[j].zero_grad()
                    errD = discriminator_loss(netsD[j], imgs[j], fake_imgs[j],
                                              sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[j].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (j, errD.data.item())

                # (4) Update G network: maximize log(D(G(z)))
                # compute total loss for training G
                #step += 1
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, caption_cnn, caption_rnn, captions, fake_imgs,
                                   real_labels, sent_emb, cap_lens)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.data.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if batch_num % 100 == 0:
                    print(D_logs + '\n' + G_logs)

                # save images
                '''
                if batch_num % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                '''

                D_losses.append(errD_total.data.item())
                G_losses.append(errG_total.data.item())

            end_t = time.time()

            print('''[%d/%d][%d]
                              Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data.item(), errG_total.data.item(),
                     end_t - start_t))

            #if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
            self.save_model(netG, avg_param_G, netsD, epoch) # Save every epoch
            self.save_losses(D_losses, G_losses, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)


    def save_losses(self, D_losses, G_losses, epoch):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()

        losses_name = 'G_D_losses_epoch_%d' % epoch
        losses_path = os.path.join(self.losses_dir, losses_name)
        plt.savefig(losses_path)


    def save_single_images(self, images, filenames, save_dir,
                           split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)


    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for model is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            # TODO: Add support for DCGAN?
            '''
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            '''
            netG = G_NET()
            netG.apply(weights_init)
            netG.to(cfg.DEVICE)
            netG.eval()

            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from : ', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.to(cfg.DEVICE)
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.to(cfg.DEVICE)

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfint('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            # TODO: See if range should be changed
            for _ in range(1): # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(tqdm(self.data_loader)):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)

                    _, captions, cap_lens = data
                    captions = captions.to(cfg.DEVICE)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    # (2) Generate fake images
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%d' % (save_dir, j)
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, k)
                        im.save(fullpath)

    # TODO: Add support for new dataset
    def gen_example(self, data_dic):
        print("data_dic: ", data_dic)
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.to(cfg.DEVICE)
            text_encoder.eval()

            # the path to save generated images
            # TODO: Add support for DCGAN?
            '''
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            '''
            netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.to(cfg.DEVICE)
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                with torch.no_grad():
                    captions = torch.from_numpy(captions)
                    cap_lens = torch.from_numpy(cap_lens)

                captions = captions.to(cfg.DEVICE)
                cap_lens = cap_lens.to(cfg.DEVICE)
                for i in range(1):  # 16
                    with torch.no_grad():
                        noise = torch.FloatTensor(batch_size, nz)
                        noise = noise.to(cfg.DEVICE)
                    # (1) Extract text embeddings
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    # (2) Generate fake images
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)







