from model import G_NET, RNN_ENCODER

import os
import pickle
import nltk
import torch


class Experimenter():
    def __init__(self, z_dim, embedding_dim, models_dir, version, data_dir, data_size, epoch):
        self.nz = z_dim
        self.embedding_dim = embedding_dim
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.data_size = data_size
        self.version = version

        # Load vocabulary
        vocab_path = os.path.join(data_dir, data_size, 'vocab.pkl')
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Load text encoder
        self.build_text_encoder()

        #self.netG = self.build_models(models_dir, data_size)


    def build_text_encoder(self):
        # Load trained text encoder model
        text_encoder = RNN_ENCODER(len(self.vocab), nhidden=self.embedding_dim)
        state_dict = torch.load(os.path.join(self.models_dir, self.data_size, 'STEM/text_encoder.pth'),
                                map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        text_encoder.eval()

        self.text_encoder = text_encoder


    def build_generator(self, epoch):
        # Load trained generator model
        netG = G_NET()
        state_dict = torch.load(
            os.path.join(self.models_dir, self.data_size, 'MirrorGAN', self.version, 'netG_epoch_%d.pth' % epoch),
            map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        netG.eval()

        return netG


    def sent_to_target_ids(self, sentences: list) -> (torch.Tensor, list):
        # Prepare sentences
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


    def generate_images(self, sentences, epoch):
        # Load generator
        netG = self.build_generator(epoch)

        targets, target_lengths = self.sent_to_target_ids(sentences)
        # reset text_encoder
        batch_size = len(sentences)
        hidden = self.text_encoder.init_hidden(batch_size)

        # words_embs: batch_size x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = self.text_encoder(targets, target_lengths, hidden)
        # words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        noise = torch.FloatTensor(batch_size, self.nz)
        mask = (targets == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        # (2) Generate fake_images
        noise.data.normal_(0, 1)
        with torch.no_grad():
            fake_imgs, attn_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

        return fake_imgs


