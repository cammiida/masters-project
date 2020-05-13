import os
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from fonts.ttf import AmaticSC
import imageio
import numpy as np
import skimage
from miscc.utils import mkdir_p


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

    '''
    hyp_sentence = []
    for word_idx in hypotheses[k]:
        hyp_sentence.append(vocab.idx2word[word_idx])

    ref_sentence = []
    for word_idx in test_references[k]:
        ref_sentence.append(vocab.idx2word[word_idx])

    print('Hypotheses: ' + " ".join(hyp_sentence))
    print('References: ' + " ".join(ref_sentence))
    '''

    hypotheses, test_references = get_sentences(hypotheses, test_references, vocab)

    fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(9,6),
                            subplot_kw={'xticks': [], 'yticks': []})

    imgs = imgs[0]
    for i in range(len(imgs)):
        ax = axs.flat[i]
        img = imgs[i]

        ax.imshow(img, interpolation='bilinear', cmap='viridis')
        ax.set_xlabel("%s\n%s" % (hypotheses[i], test_references[i]), fontsize=6, wrap=True)
        #ax.set_title(hypotheses[i], wrap=True, fontdict={'fontsize': 6, 'fontweight': 'medium'})

    plt.tight_layout()
    plt.show()

    img_dim = 336  # 14*24
    img = imgs[0][k]
    imageio.imwrite('img.jpg', img)



    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    #fnt = ImageFont.truetype(AmaticSC)
    # get a drawing contextF
    #d = ImageDraw.Draw(img_txt)
    '''
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
    try:
        plt.show()
    except RuntimeError:
        print("Could not show image.")
    '''
