import torch
import pickle
from transformers import AlbertTokenizer, AlbertModel, BertTokenizer, BertModel

albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


def test_albert_tokenization(text):
    tokenized_text = albert_tokenizer.tokenize(text)
    print('ALBERT: ', tokenized_text)

    # Convert token to vocabulary indices
    indexed_tokens = albert_tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensors

def test_bert_tokenization(text):
    tokenized_text = bert_tokenizer.tokenize(text)
    print('BERT: ', tokenized_text)

    # Convert token to vocabulary indices
    indexed_tokens = albert_tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_captions, _ = model(tokens_tensor, segments_tensors)

    # Load vocabulary
    with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    for cap_idx in encoded_captions:
        print(cap_idx)
        while len(cap_idx) < 12:
            cap_idx.append(0)

        cap = ' '.join([vocab.idx2word[word_idx.item()] for word_idx in cap_idx])
        cap = u'[CLS] ' + cap

        tokenized_cap = bert_tokenizer.tokenize(cap)
        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_cap)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor)


    return encoded_layers

def test_albert_embedding_squeeze(tokens_tensor, segments_tensors):
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    return encoded_layers

def test_bert_embedding(tokens_tensor, segments_tensors):
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)


if __name__ == '__main__':
    encoded_layers = test_albert_tokenization("[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]")
    #encoded_layers = test_albert_embedding_squeeze(tokens_tensor, segments_tensors)
    print(encoded_layers)
    print(encoded_layers[1])
    print(encoded_layers[1].squeeze(0))



