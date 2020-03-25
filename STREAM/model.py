import pickle
import torch
import torch.nn as nn
import torchvision.models as models

from cfg.config import cfg, cfg_from_file
from transformers import AlbertTokenizer, AlbertModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################
# Encoder RESNET CNN
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
    def __init__(self, vocab, use_glove, use_albert):
        super(Decoder, self).__init__()
        self.vocab = vocab
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
        self.vocab_size = len(vocab)
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
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            # load Glove embeddings
            if use_glove:
                glove_vectors = pickle.load(open('../data/glove.6B/glove_words.pkl', 'rb'))
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

                cap = ' '.join([self.vocab.idx2word[word_idx.item()] for word_idx in cap_idx])
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
