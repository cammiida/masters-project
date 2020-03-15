import os
import json
import pickle
from STREAM.processData import Vocabulary


def check_json(path):
    with open(path, 'rb') as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            print(key)
            if isinstance(data[key], list):
                print(data[key][0])


def check_pickle(path):
    file_size = os.path.getsize(path)
    print("Size of file %s: %s" % (path, str(file_size)))
    if file_size > 0:
        with open(path, 'rb') as f:
            x = pickle.load(f, encoding='utf-8')
            print(x)
            print(type(x))
    else:
        print("File is empty")


def check_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f.read())


def json_matching_files(json_path, files_dir):
    both = []
    only_caption = []
    only_file = []

    with open(json_path, 'rb') as f:
        captions = json.load(f)

    files = os.listdir(files_dir)
    images = captions["images"]
    file_names = [image['file_name'] for image in images]


    for i in range(0, max(len(files), len(file_names))):
        if file_names[i] in files:
            both.append(file_names[i])
        else:
            only_caption.append(file_names[i])
        if files[i] in file_names:
            both.append(files[i])
        else:
            only_file.append(files[i])

    print("both: \nLength: ", len(both), "list: ", both)
    print("only caption: \nLength: ", len(only_caption), "list: ", only_caption)
    print("only file: \nLength: ", len(only_file), "list: ", only_file)


if __name__ == '__main__':
    big_root = '../data/big'
    small_root = '../data/small'

    check_json(os.path.join(big_root, 'annotations/instances_train2014.json'))
    check_json(os.path.join(small_root, 'annotations/captions_train2014.json'))
    # check_pickle(os.path.join(big_root, 'captions.pickle'))
    # check_text_file(os.path.join(big_root, 'example_captions.txt'))
    # check_text_file(os.path.join(big_root, 'example_filenames.txt'))

    #json_matching_files(os.path.join(small_root, 'annotations/captions_train2014.json'),
    #                    os.path.join(small_root, 'train2014'))