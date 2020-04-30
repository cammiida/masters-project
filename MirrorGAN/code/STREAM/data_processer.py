from cfg.config import cfg
from model import Encoder, Decoder
from datasets import Vocabulary

import os
import pickle
import torch
import nltk
from collections import Counter
from PIL import Image
from pycocotools.coco import COCO
from datetime import datetime
from nltk.translate.bleu_score import corpus_bleu


#############
# Init model
#############
def init_model(vocabulary):

    encoder = Encoder().to(cfg.DEVICE)
    decoder = Decoder(vocab=vocabulary).to(cfg.DEVICE)

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


def process_data(caption_path, vocab_path, threshold):
    # Create and save vocab
    vocab = build_vocab(json=caption_path, threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("resizing images...")
    splits = ['val', 'train']

    for split in splits:
        folder = os.path.join(cfg.DATA_DIR, '%s2014' % split)
        resized_folder = os.path.join(cfg.DATA_DIR, '%s2014_resized/' % split)
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, image_file), image.format)

            if i % 1000 == 0 or i == num_images:
                print("copied %s/%s images" % (str(i), str(num_images)))

        print("copied %s images in %s folder..." % (str(num_images), split))

    print("done resizing images...")


def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        # TODO: Can this be done with BERT tokenizer?
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokens = tokenizer.tokenize(caption.lower())
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # TODO: Remove this?
    # omit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('<unk>') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def resize_image(image):
    width, height = image.size
    # Make square based on the smallest dim
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    # Downsample with high quality (antialias)
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image


def write_results(vocab, hypotheses, references):
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

# TODO: Fix this to run? Or just write hypotheses to file
def print_sample(hypotheses, references, test_references, imgs, alphas, k, show_att, losses, vocab):
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