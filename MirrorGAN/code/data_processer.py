# From STREAM
from pycocotools.coco import COCO
from collections import Counter
import nltk
from transformers import BertTokenizer, BertModel


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        # TODO: Can this be done with BERT tokenizer?
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


if __name__ == '__main__':
    ann_path = '../../data/small/annotations/captions_train2014.json'
    thresh = 5
    build_vocab(ann_path, threshold=thresh)