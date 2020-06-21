import os
import json
import pickle
from random import randint
from tqdm import tqdm
import copy

def check_json(path):
    with open(path, 'rb') as json_file:
        data = json.load(json_file)
        print("num images: ", len(data["images"]))
        for key, value in data.items():
            print(key)
            if isinstance(data[key], list):
                print(data[key][0:2])


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

def check_caption_val2014():
    caption_file_path = '../data/small1/annotations/captions_val2014.json'
    with open(caption_file_path, 'rb') as f:
        captions_dict = json.load(f, encoding='utf-8')

    images = captions_dict["images"]
    annotations = captions_dict["annotations"]

    idx = randint(0, len(images) - 1)
    img_id = images[idx]["id"]
    filename = images[idx]["file_name"].replace('.jpg', '')
    print("filename: ", filename)
    print("img_id: ", img_id)

    ann_dict = next(ann for ann in annotations if ann["image_id"] == img_id)
    print(ann_dict)
    cap = ann_dict["caption"]
    print("Caption: ", cap)


def check_create_json(old_root, new_root):
    keys = {
        "images": {"check_key": "file_name", "del_key": "id", "extension": ".jpg"},
        "annotations": {"check_key": "image_id", "del_key": "category_id", "extension": ""}
    }
    dir_name = 'annotations'
    json_name = 'captions_val2014.json'
    old_json_path = os.path.join(old_root, dir_name, json_name)
    new_json_path = os.path.join(new_root, dir_name, json_name)
    tqdm.write("Creating new json file %s from %s..." % (new_json_path, old_json_path))

    with open(old_json_path, 'rb') as old_json:
        data = json.load(old_json)
        new_json = copy.deepcopy(data)

    lengths = dict()
    del_list = []
    for key, value in tqdm(keys.items()):
        lengths[key] = len(new_json[key])
        tqdm.write("key: %s" % key)
        tqdm.write("value: %s" % value)
        tqdm.write("value['check_key']: %s" % value["check_key"])
        tqdm.write("randint: %d" % randint(0, len(new_json["images"])-1))


if __name__ == '__main__':
    big_root = '../data/big'
    small_root = '../data/small'

    #check_create_json(big_root, small_root)
    #check_caption_val2014()
    #check_pickle(os.path.join(small_root, 'train/filenames.pickle'))
    #check_pickle(os.path.join(small_root, 'test/filenames.pickle'))

    # check_json(os.path.join(big_root, 'annotations/instances_train2014.json'))
    # check_json(os.path.join(big_root, 'annotations/instances_val2014.json'))
    # check_pickle(os.path.join(big_root, 'captions.pickle'))
    # check_text_file(os.path.join(big_root, 'example_captions.txt'))
    # check_text_file(os.path.join(big_root, 'example_filenames.txt'))

    #json_matching_files(os.path.join(small_root, 'annotations/captions_train2014.json'),
    #                    os.path.join(small_root, 'train2014'))
    print(len(os.listdir('../data/big/text/')))
    print(len(os.listdir('../data/small/text/')))