import pickle
import torch.nn as nn
import torch
from transformers import AlbertTokenizer, AlbertModel, AlbertForMaskedLM, BertTokenizer, BertModel, BertForMaskedLM
import imageio
from torchtext.vocab import Vectors, GloVe
import torchvision.models as models


###################
# START Parameters
###################

# hyperparams
grad_clip = 5.
num_epochs = 4
batch_size = 32
decoder_lr = 0.0004

# if both are false them model = baseline

glove_model = False
bert_model = False

from_checkpoint = True
train_model = False
valid_model = True
eval = True

###################
# END Parameters
###################

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model tokenizer (vocabulary)
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = AlbertModel.from_pretrained('albert-base-v2')

# Load GloVe
glove_vectors = pickle.load(open('glove.6B/glove_words.pkl', 'rb'))
glove_vectors = torch.tensor(glove_vectors)


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
    def __init__(self, vocab_size, use_glove, use_bert):
        super(Decoder, self).__init__()
        self.encoder_dim = 2048
        self.attention_dim = 512
        self.use_bert = use_bert

        if use_glove:
            self.embed_dim = 300
        elif use_bert:
            self.embed_dim = 768
        else:
            self.embed_dim = 512

        self.decoder_dim = 512
        self.vocab_size = vocab_size
        self.dropout = 0.5

        # soft attention
        self.enc_att = nn.Linear(2048, 512)
        self.dec_att = nn.Linear(512, 512)
        self.att = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # decoder layers
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        if not use_bert:
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            # load Glove embeddings
            if use_glove:
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

        # load bert or regular embeddings
        if not self.use_bert:
            embeddings = self.embedding(encoded_captions)
        elif self.use_bert:
            embeddings = []
            for cap_idx in encoded_captions:
                # padd caption to correct size
                while len(cap_idx) < max_dec_len:
                    cap_idx.append(PAD)

                cap = ' '.join([vocab.idx2word[word_idx.item()] for word_idx in cap_idx])
                cap = u'[CLS] '+cap

                tokenized_cap = tokenizer.tokenize(cap)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_cap)
                tokens_tensor = torch.tensor([indexed_tokens]).to(device)

                with torch.no_grad():
                    encoded_layers, _ = AlbertModel(tokens_tensor)

                # Remove all dims with dim_size = 0 in dim 11
                albert_embedding = encoded_layers[11].squeeze(0)

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
                        if token == full_token and curr_token == '':
                            tokens_embedding.append(piece_embedding)
                            j += 1
                            break
                        else: # partial token
                            x += 1

                            if curr_token == '':
                                tokens_embedding.append(piece_embedding)
                                curr_token += token.replace('#', '')
                            else:
                                tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                                curr_token += token.replace('#', '')

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


# vocab indices
PAD = 0
START = 1
END = 2
UNK = 3

# Load vocabulary
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)