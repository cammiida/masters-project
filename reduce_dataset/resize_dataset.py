import os
import pickle
import random
import math
import json
import copy
from shutil import copyfile
from random import randint
from tqdm import tqdm, trange

######################
#   HELPER METHODS   #
######################
def delete_from_dict(obj_dict, obj_key, check_key, control_list, extension='', del_key=None):
    i = 0
    count = 1
    del_list = []

    objects = obj_dict[obj_key]
    org_len = len(objects)
    while i < len(objects):

        obj = objects[i]
        value = obj[check_key]
        if isinstance(value, str): value = value.replace(extension, '')

        if value not in control_list:
            if del_key is not None: del_list.append(obj[del_key])
            objects.pop(i)
        else:
            i += 1

        count += 1
        if count % 1000 == 0 or count == org_len:
            print("Number of ", obj_key, " checked: ", count, "/", org_len )


    return obj_dict, del_list if len(del_list) > 0 else obj_dict


def load_pickle(path):
    with open(path, 'rb') as f:
        filenames = pickle.load(f, encoding='utf-8')
        print(filenames[:10])
    return filenames


def save_pickle(data, directory, filename):
    if not os.path.isdir(directory):
        create_dir(directory)

    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_json(path):
    with open(path, 'rb') as f:
        data = json.load(f, encoding='utf-8')

    return data


# percentile < 1
def generate_sample(filenames, percentile):
    assert percentile < 1
    k = math.floor(len(filenames) * percentile)
    sample = random.sample(filenames, k)
    return sample


def check_json(file):
    with open(file, 'rb') as json_file:
        data = json.load(json_file)
        print(data["categories"][:10])
        for key, value in data.items():
            print("Key: ", key)

def create_dir(path):
    print("create dir: %s" % path)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

##########################
#   END HELPER METHODS   #
##########################

def get_sample_filenames(data_dir, percentile):
    print("Getting sample filenames...")
    train_names = load_pickle(os.path.join(data_dir, 'train', 'filenames.pickle'))
    test_names = load_pickle(os.path.join(data_dir, 'test', 'filenames.pickle'))

    sample_train_names = generate_sample(train_names, percentile)
    sample_test_names = generate_sample(test_names, percentile)

    return sample_train_names, sample_test_names


def save_samples(destination, train_sample, test_sample):
    save_pickle(train_sample, os.path.join(destination, 'train'), 'filenames.pickle')
    save_pickle(test_sample, os.path.join(destination, 'test'), 'filenames.pickle')


def copy_files(source, destination, filenames, extension):
    print("Copying files from %s to %s..." % (source, destination))

    files = os.listdir(source)
    for i, f in enumerate(tqdm(files)):
        if f.replace(extension, '') in filenames:
            src = os.path.join(source, f)
            dst = os.path.join(destination, f)

            copyfile(src, dst)

            #if os.path.isfile(dst):
            #    print("Copied file %s to %s." % (src, dst))
            if not os.path.isfile(dst):
                print("Could not copy file %s to %s." % (src, dst))


def create_json(old_json_path, new_json_path, filenames, keys):
    print("Creating new json file %s from %s..." % (old_json_path, new_json_path))

    with open(old_json_path, 'rb') as old_json:
        data = json.load(old_json)
        new_json = copy.deepcopy(data)

    lengths = dict()
    for key, value in tqdm(keys.items()):
        lengths[key] = len(new_json[key])

        new_json, del_list = delete_from_dict(obj_dict=new_json, obj_key=key, check_key=value["check_key"],
                                              control_list=filenames, extension=value["extension"], del_key=value["del_key"])

    for key, value in keys.items():
        print("Length of %s before start: %s" % (key, str(lengths[key])))
        print("Length of %s after run: %s" % (key, str(len(new_json[key]))))

    with open(new_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_json, outfile)

        print("Saved json to: ", new_json_path)


def create_annotations(source, destination, train_sample, test_sample):
    keys = {
        "images": {"check_key": "file_name", "del_key": "id", "extension": ".jpg"},
        "annotations": {"check_key": "image_id", "del_key": "category_id", "extension": ""},
        "categories": {"check_key": "id", "del_key": None, "extension": ""}
    }

    keys_without_cat = copy.deepcopy(keys)
    del keys_without_cat["categories"]

    create_json(os.path.join(source, 'annotations/captions_train2014.json'),
                os.path.join(destination, 'annotations/captions_train2014.json'),
                train_sample, keys_without_cat)
    create_json(os.path.join(source, 'annotations/captions_val2014.json'),
                os.path.join(destination, 'annotations/captions_val2014.json'),
                test_sample, keys_without_cat)

    create_json(os.path.join(source, 'annotations/instances_train2014.json'),
                os.path.join(destination, 'annotations/instances_train2014.json'),
                train_sample, keys)
    create_json(os.path.join(source, 'annotations/instances_val2014.json'),
                os.path.join(destination, 'annotations/instances_val2014.json'),
                test_sample, keys)
    create_json(os.path.join(source, 'annotations/person_keypoints_train2014.json'),
                os.path.join(destination, 'annotations/person_keypoints_train2014.json'),
                train_sample, keys)

    create_json(os.path.join(source, 'annotations/person_keypoints_val2014.json'),
                os.path.join(destination, 'annotations/person_keypoints_val2014.json'),
                test_sample, keys)


def gen_ex_fnames_n_caps(caption_file_path, example_filenames_dest, example_captions_dest, filenames, n_files=9):
    captions_dict = load_json(caption_file_path)
    images = captions_dict["images"]
    annotations = captions_dict["annotations"]

    example_filenames_file = open(example_filenames_dest, 'w', encoding='utf-8')
    example_captions_file = open(example_captions_dest, 'w', encoding='uft-8')

    example_filenames_file.write("interpolate_captions\n")
    example_filenames_file.write("example_captions\n")

    for i in trange(n_files):
        idx = randint(0, len(images)-1)
        img_id = images[idx]["id"]
        # Find corresponding caption
        ann_dict = next(ann for ann in annotations if ann["image_id"] == img_id)
        caption = ann_dict["caption"]

        filename = filenames.pop(img_id)
        example_filenames_file.write("text/%s\n" % filename)
        example_captions_file.write("%s\n" % caption)

    print("Done creating %s and %s" % (example_filenames_dest, example_captions_dest))


def copy_text_files(source, destination, train_test_filenames):
    copy_files(source=os.path.join(source, 'text'),
               destination=os.path.join(destination, 'text'),
               extension='.txt',
               filenames=train_test_filenames)


def copy_images(source, destination, folders, train_sample, test_sample):
    filenames=[]
    for folder_name in tqdm(folders):
        if "train" in folder_name:
            filenames = train_sample
        elif "val" in folder_name:
            filenames = test_sample

        copy_files(source=os.path.join(source, folder_name),
                   destination=os.path.join(destination, folder_name),
                   extension='.jpg',
                   filenames=filenames)

def main(big_root, small_root):
    # GET SAMPLE OF FILENAMES FROM big_root AND SAVE THEM TO small_root
    percentage = 0.05
    train_filenames, test_filenames = get_sample_filenames(big_root, percentage)
    save_samples(small_root, train_filenames, test_filenames)

    # GENERATE ALL FILES IN ANNOTATIONS AND FILTER OUT ALL ENTRIES THAT DO NOT CORRESPOND TO THE SAMPLE FILENAMES
    create_annotations(source=big_root,
                       destination=small_root,
                       train_sample=train_filenames,
                       test_sample=test_filenames)

    # Generate example_captions.txt, example_filenames.txt
    filenames_dest = os.path.join(small_root, 'example_filenames.txt')
    captions_dest = os.path.join(small_root, 'example_captions.txt')
    cap_path = os.path.join(small_root, 'annotations/captions_val2014.json')
    gen_ex_fnames_n_caps(caption_file_path=cap_path,
                         example_filenames_dest=filenames_dest,
                         example_captions_dest=captions_dest,
                         filenames=test_filenames)

    # COPY TEXT FILES CORRESPONDING TO THE SAMPLE FILENAMES FROM big_root TO small_root
    train_test_filenames = train_filenames + test_filenames
    copy_files(source=os.path.join(big_root, 'text'),
               destination=os.path.join(small_root, 'text'),
               extension='.txt',
               filenames=train_test_filenames)

    # COPY IMAGES CORRESPONDING TO THE SAMPLE FILENAMES FROM big_root TO small_root
    # COPY IMAGES IN train2014/ and val2014
    folder_names = ["train2014", "val2014", "train2014_resized", "val2014_resized"]
    copy_images(source=big_root,
                destination=small_root,
                folders=folder_names,
                train_sample=train_filenames,
                test_sample=test_filenames)


if __name__ == '__main__':
    big_root = '../data/big'
    small_root = '../data/small'

    main(big_root, small_root)










