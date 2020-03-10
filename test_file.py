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

def test_bert_tokenization(cap):
    tokenized_cap = bert_tokenizer.tokenize(cap)
    print('cap: ', cap)
    print('tokenized_cap: ', tokenized_cap)


    # Convert token to vocabulary indices
    indexed_tokens = albert_tokenizer.convert_tokens_to_ids(tokenized_cap)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)

    print('encoded_layers: ', encoded_layers)
    print(encoded_layers.size())
    print(encoded_layers.squeeze(0).size())
    bert_embedding = encoded_layers.squeeze(0)

    print('bert_embedding: ', bert_embedding)

    embeddings = []

    split_cap = cap.split()
    tokens_embedding = []
    j = 0

    for full_token in split_cap:
        curr_token = ''
        x = 0
        print('full_token: ', full_token)
        for i,_ in enumerate(tokenized_cap[1:]): # disregard CLS
            #print('i, j: ', i, j)
            token = tokenized_cap[i+j]
            piece_embedding = bert_embedding[i+j]
            # print('token: ', token)
            # print('piece_embedding: ', piece_embedding)

            # full token
            if token == full_token and curr_token == '':
                tokens_embedding.append(piece_embedding)
                j += 1
            else: # partial token
                x += 1
                if curr_token == '':
                    tokens_embedding.append(piece_embedding)
                    curr_token += token.replace('#', '')
                else:
                    tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                    curr_token += token.replace('#', '')

                    if curr_token == full_token: # end of partial
                        j += x
                        break

    cap_embeddings = torch.stack(tokens_embedding)
    print('cap_embeddings size: ', cap_embeddings.size())
    embeddings.append(cap_embeddings)

    embeddings = torch.stack(embeddings)

    print('embeddings: ', embeddings)



def test_albert_embedding_squeeze(tokens_tensor, segments_tensors):
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    return encoded_layers

def test_bert_embedding(tokens_tensor, segments_tensors):
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)


def test_encode_plus(text_a, text_b):
    padded_sequence_a = albert_tokenizer.encode_plus(text_a, max_length=19, pad_to_max_length=True)
    padded_sequence_b = albert_tokenizer.encode_plus(text_b, max_length=19, pad_to_max_length=True)
    padded_sequence = albert_tokenizer.encode_plus(text_a, text_b, max_length=19, pad_to_max_length=True)
    print('padded_sequence_a: ', padded_sequence_a)
    print('padded_sequence_b: ', padded_sequence_b)
    print('padded_sequences: ', padded_sequence)

def text_encoded_sequence(text_a, text_b):
    encoded_sequence = albert_tokenizer.encode(text_a, text_b)
    print(albert_tokenizer.decode(encoded_sequence ))

    return albert_tokenizer.decode(encoded_sequence)

if __name__ == '__main__':
    text_a = "Who was Jim Henson ?"
    text_b = "Jim Henson was a puppeteer"
    # test_bert_tokenization("[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]")
    # encoded_layers = test_albert_embedding_squeeze(tokens_tensor, segments_tensors)
    test_encode_plus(text_a, text_b)
    # text_encoded_sequence(text_a, text_b)



