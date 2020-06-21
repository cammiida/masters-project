import os
import pickle
import random
import math
import json
import copy
from shutil import copyfile
from random import randint
from tqdm import tqdm

######################
#   HELPER METHODS   #
######################
def filter_dict(obj_dict, obj_key, check_key, control_list, extension='', control_key=None):
    i = 0
    # count = 1
    new_ctrl_list = []

    objects = obj_dict[obj_key] # e.g. annotations
    # org_len = len(objects) # len(annotations)
    while i < len(objects):
        obj = objects[i] # e.g. annotation
        value = obj[check_key] # for annotations: image_id
        if isinstance(value, str): value = value.replace(extension, '')

        if value in control_list: # list of image ids for annotations
            # if there is a control key, add value of control_key to new control list and increment i
            if control_key is not None: new_ctrl_list.append(obj[control_key])
            i += 1
        # if file_name was not in filenames list, remove it from list and don't increment i
        else:
            objects.pop(i)
        '''

        count += 1
        if count % 1000 == 0 or count == org_len:
            tqdm.write("Number of %s checked: %d/%d" % (obj_key, count, org_len) )
        '''
    return obj_dict, new_ctrl_list


def load_pickle(directory, filename):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    path = os.path.join(directory, filename)
    with open(path, 'rb') as f:
        filenames = pickle.load(f, encoding='utf-8')
    return filenames


def save_pickle(data, directory, filename):
    if not os.path.isdir(directory):
        create_dir(directory)

    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        tqdm.write("Pickle already exists! Deleting...")
        os.remove(file_path)
    tqdm.write("Saving new pickle...")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    tqdm.write("New pickle saved to %s!" % file_path)

def load_json(path):
    with open(path, 'rb') as f:
        data = json.load(f)

    return data


def get_train_test_filenames():
    train_dir = os.path.join(small_root, 'train')
    test_dir = os.path.join(small_root, 'test')
    file_name = 'filenames.pickle'

    train_filenames = load_pickle(train_dir, file_name)
    test_filenames = load_pickle(test_dir, file_name)

    return train_filenames, test_filenames


# percentile < 1
def generate_sample(filenames, percentile):
    # Get filenames
    assert percentile < 1
    k = math.floor(len(filenames) * percentile)
    sample = random.sample(filenames, k)
    return sample


def check_json(file):
    with open(file, 'rb') as json_file:
        data = json.load(json_file)
        tqdm.write(data["categories"][:10])
        for key, value in data.items():
            tqdm.write("Key: ", key)


def create_dir(path):
    tqdm.write("create dir: %s" % path)
    try:
        os.mkdir(path)
    except OSError:
        tqdm.write("Creation of the directory %s failed" % path)
    else:
        tqdm.write("Successfully created the directory %s" % path)


##########################
#   END HELPER METHODS   #
##########################

def generate_sample_filenames(percentile):
    tqdm.write("Getting sample filenames...")
    train_dir = os.path.join(big_root, 'train')
    test_dir = os.path.join(big_root, 'test')
    file_name = 'filenames.pickle'
    train_filenames = load_pickle(train_dir, file_name)
    test_filenames = load_pickle(test_dir, file_name)

    sample_train_filenames = generate_sample(train_filenames, percentile)
    sample_test_filenames = generate_sample(test_filenames, percentile)

    return sample_train_filenames, sample_test_filenames


def save_samples(train_sample, test_sample):
    tqdm.write("Saving sample filenames in filenames.pickle...")
    save_pickle(train_sample, os.path.join(small_root, 'train'), 'filenames.pickle')
    save_pickle(test_sample, os.path.join(small_root, 'test'), 'filenames.pickle')


def copy_files(filenames, extension):
    tqdm.write("Copying files from %s to %s..." % (big_root, small_root))

    files = os.listdir(big_root)
    for i, f in enumerate(tqdm(files)):
        if f.replace(extension, '') in filenames:
            src = os.path.join(big_root, f)
            dst = os.path.join(small_root, f)

            copyfile(src, dst)

            if not os.path.isfile(dst):
                tqdm.write("Could not copy file %s to %s." % (src, dst))


def create_json(json_name, filenames, keys):
    dir_name = 'annotations'
    old_json_path = os.path.join(big_root, dir_name, json_name)
    new_json_path = os.path.join(small_root, dir_name, json_name)
    tqdm.write("Creating new json file %s from %s..." % (new_json_path, old_json_path))

    with open(old_json_path, 'rb') as old_json:
        data = json.load(old_json)
        new_json = copy.deepcopy(data)

    lengths = dict()
    control_list = []
    for key, value in tqdm(keys.items()):
        lengths[key] = len(new_json[key])

        new_json, control_list = filter_dict(obj_dict=new_json,
                                         obj_key=key,
                                         check_key=value["check_key"],
                                         control_list=filenames if key=="images" else control_list,
                                         extension=value["extension"],
                                         control_key=value["control_key"])
    tqdm.write("new_json after all deletion: %s" % new_json)

    for key, value in keys.items():
        tqdm.write("Length of %s before start: %s" % (key, str(lengths[key])))
        tqdm.write("Length of %s after run: %s" % (key, str(len(new_json[key]))))

    new_dir_path = os.path.join(small_root, dir_name)
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)
    if os.path.isfile(new_json_path):
        os.remove(new_json_path)
    with open(new_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_json, outfile)

        tqdm.write("Saved json to: %s " % new_json_path)


def create_annotations():
    # Get filenames from filenames.pickle
    train_sample, test_sample = get_train_test_filenames()

    tqdm.write("Train sample %s " % train_sample)
    tqdm.write("Test sample %s" % test_sample)
    tqdm.write("Creating annotations...")
    keys = {
        "images": {"check_key": "file_name", "control_key": "id", "extension": ".jpg"},
        "annotations": {"check_key": "image_id", "control_key": "category_id", "extension": ""},
        "categories": {"check_key": "id", "control_key": None, "extension": ""}
    }

    keys_without_cat = copy.deepcopy(keys)
    keys_without_cat["annotations"]["control_key"] = None
    del keys_without_cat["categories"]

    json_files = [{"json_name": "captions_train2014.json", "keys": keys_without_cat, "filenames": train_sample},
                  {"json_name": "captions_val2014.json", "keys": keys_without_cat, "filenames": test_sample},
                  {"json_name": "instances_train2014.json", "keys": keys, "filenames": train_sample},
                  {"json_name": "instances_val2014.json", "keys": keys, "filenames": test_sample},
                  {"json_name": "person_keypoints_train2014.json", "keys": keys, "filenames": train_sample},
                  {"json_name": "person_keypoints_val2014.json", "keys": keys, "filenames": test_sample}]

    for json_file in json_files:
        create_json(json_name=json_file["json_name"], filenames=json_file["filenames"], keys=json_file["keys"])


def gen_ex_fnames_n_caps(n_caps=9):
    example_filenames_dest = os.path.join(small_root, 'example_filenames.txt')
    example_captions_dest = os.path.join(small_root, 'example_captions.txt')
    caption_file_path = os.path.join(small_root, 'annotations/captions_val2014.json')

    tqdm.write("Generating example filenames and captions...")
    captions_dict = load_json(caption_file_path)
    images = captions_dict["images"]
    annotations = captions_dict["annotations"]

    # Remove files if they already exist
    if os.path.isfile(example_filenames_dest):
        os.remove(example_captions_dest)
    if os.path.isfile(example_captions_dest):
        os.remove(example_captions_dest)

    example_filenames_file = open(example_filenames_dest, 'w', encoding='utf-8')
    example_captions_file = open(example_captions_dest, 'w', encoding='utf-8')

    example_filenames_file.write("interpolate_captions\n")
    example_filenames_file.write("example_captions\n")

    i = 0
    while i < n_caps:
        idx = randint(0, len(images)-1)
        img_id = images[idx]["id"]
        filename = images[idx]["file_name"].replace('.jpg', '')
        # Find corresponding caption
        try:
            ann_dict = next(ann for ann in annotations if ann["image_id"] == img_id)
            caption = ann_dict["caption"]
            example_filenames_file.write("text/%s\n" % filename)
            example_captions_file.write("%s\n" % caption)
            i += 1
            tqdm.write("Found corresponding caption for image with id %s" % str(img_id))
        except StopIteration:
            tqdm.write("Could not find corresponding caption for image with id %s" % str(img_id))

    example_filenames_file.close()
    example_captions_file.close()

    tqdm.write("Done creating %s and %s" % (example_filenames_dest, example_captions_dest))


def copy_text_files():
    tqdm.write("Copying text files...")
    # Get the copied files' names
    train_filenames, test_filenames = get_train_test_filenames()
    train_test_filenames = train_filenames + test_filenames

    copy_files(extension='.txt', filenames=train_test_filenames)


def copy_images(folders):
    tqdm.write("Copying images...")
    # Get the copied files' names
    train_filenames, test_filenames = get_train_test_filenames()

    filenames=[]
    for folder_name in tqdm(folders):
        if "train" in folder_name:
            filenames = train_filenames
        elif "val" in folder_name:
            filenames = test_filenames

        copy_files(extension='.jpg', filenames=filenames)

def main():
    # GET SAMPLE OF FILENAMES FROM big_root AND SAVE THEM TO small_root
    train_filenames, test_filenames = generate_sample_filenames(percentage)
    save_samples(train_filenames, test_filenames)

    # GENERATE ALL FILES IN ANNOTATIONS AND FILTER OUT ALL ENTRIES THAT DO NOT CORRESPOND TO THE SAMPLE FILENAMES
    create_annotations()

    # Generate example_captions.txt, example_filenames.txt
    gen_ex_fnames_n_caps()

    # COPY TEXT FILES CORRESPONDING TO THE SAMPLE FILENAMES FROM big_root TO small_root
    copy_text_files()

    # COPY IMAGES CORRESPONDING TO THE SAMPLE FILENAMES FROM big_root TO small_root
    # COPY IMAGES IN train2014/ and val2014
    folder_names = ["train2014", "val2014", "train2014_resized", "val2014_resized"]
    copy_images(folders=folder_names)


def test_create_annotations():
    filenames = ["COCO_val2014_000000391895"]
    test_json = {
        "images": [{"id": 391895, "file_name": "COCO_val2014_000000391895"},
                   {"id": 522418, "file_name": "COCO_val2014_000000522418"}],
        "annotations": [{"image_id": 391895, "id": 156, "category_id": 1},
                        {"image_id": 522418, "id": 509, "category_id": 2}],
        "categories": [{"id": 1}, {"id": 2}]
    }

    new_json, control_list = filter_dict(test_json, "images", check_key="file_name", control_list=filenames,
                                     extension=".jpg", control_key="id")
    print("new_json: ", new_json)
    print("control_list: ", control_list)
    new_json, control_list = filter_dict(new_json, "annotations", check_key="image_id", control_list=control_list,
                                     control_key="category_id")
    print("new_json: ", new_json)
    print("control_list: ", control_list)
    new_json, control_list = filter_dict(new_json, "categories", check_key="id", control_list=control_list)
    print("new_json: ", new_json)
    print("control_list: ", control_list)

if __name__ == '__main__':

    # PARAMETERS
    big_root = '../data/big'
    small_root = '../data/small'
    percentage = 0.05

    main()




