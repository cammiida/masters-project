from model import G_NET, RNN_ENCODER
import nltk
import torch


class Experimenter():
    def __init__(self, embedding_dim, net_E, n_words, wordtoix):
        self.embedding_dim = embedding_dim
        self.net_E = net_E
        self.n_words = n_words
        self.wordtoix = wordtoix
        print("dataset n_words: ", n_words)

        # Load text encoder
        self.build_text_encoder()


    def build_text_encoder(self):
        # Load trained text encoder model
        text_encoder = RNN_ENCODER(self.n_words, nhidden=self.embedding_dim)
        state_dict = torch.load(self.net_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        text_encoder.eval()

        self.text_encoder = text_encoder


    def build_generator(self, net_G):
        # Load trained generator model
        netG = G_NET()
        state_dict = torch.load(net_G, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        netG.eval()

        return netG


    def sent_to_target_ids(self, sentences: list) -> (torch.Tensor, torch.Tensor):
        # Prepare sentences
        tokenized_sentences = []
        for sent in sentences:
            tokens = nltk.tokenize.word_tokenize(str(sent).lower())
            caption = list()
            #caption.append(self.wordtoix['<start>'])
            for token in tokens:
                if token in self.wordtoix:
                    caption.append(self.wordtoix[token])
                #else:
                #    caption.append(self.wordtoix['<unk>'])
            #caption.extend([self.wordtoix[token] for token in tokens])
            caption.append(self.wordtoix['<end>'])
            target = torch.Tensor(caption)
            tokenized_sentences.append(target)

        # Sort longest to shortest sentence
        tokenized_sentences.sort(reverse=True, key=len)

        lengths = [len(sent) for sent in tokenized_sentences]

        targets = torch.zeros(len(tokenized_sentences), max(lengths)).long()
        for i, cap in enumerate(tokenized_sentences):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        lengths = torch.tensor(lengths)

        return targets, lengths


    def generate_images(self, net_G, z_dim, sentences=None, sentence_embeddings=None, sentence_lengths=None):
        # Load generator
        netG = self.build_generator(net_G)

        if sentences:
            targets, target_lengths = self.sent_to_target_ids(sentences)
            batch_size = len(sentences)
        elif sentence_embeddings is not None and sentence_lengths is not None:
            targets, target_lengths = sentence_embeddings, sentence_lengths
            batch_size = len(sentence_embeddings)
        else:
            raise RuntimeError('Cannot generate images based on nothing!')
        # reset text_encoder

        with torch.no_grad():
            hidden = self.text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = self.text_encoder(targets, target_lengths, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            noise = torch.FloatTensor(batch_size, z_dim)
            mask = (targets == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]

            # (2) Generate fake_images
            noise.data.normal_(0, 1)
            fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)

        return fake_imgs





