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
from tqdm import tqdm


# TODO: Fix paths
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


    write_results(vocab, hypotheses, test_references)

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