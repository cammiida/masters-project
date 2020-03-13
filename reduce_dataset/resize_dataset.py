import os
import pickle
import random
import math
import json
import copy
from shutil import copyfile



def load_pickle(path):
    with open(path, 'rb') as f:
        filenames = pickle.load(f, encoding='utf-8')
        print(filenames[:10])
    return filenames


# percentile < 1
def get_filenames_sample(filenames, percentile):
    assert percentile < 1
    k = math.floor(len(filenames) * percentile)
    sample = random.sample(filenames, k)
    return sample


def create_pickle(lst, directory, filename):
    # TODO: Check that file doesn't exist
    print("%s is directory: %s" % (directory, os.path.isdir(directory)))
    if not os.path.isdir(directory):
        create_dir(directory)
    with open(os.path.join(directory, filename), 'wb') as f:
        pickle.dump(lst, f)



def create_dir(path):
    print("create dir: %s" % path)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)


def get_sample_filenames(data_dir, percentile):
    train_names = load_pickle(os.path.join(data_dir, 'train', 'filenames.pickle'))
    test_names = load_pickle(os.path.join(data_dir, 'test', 'filenames.pickle'))

    sample_train_names = get_filenames_sample(train_names, percentile)
    sample_test_names = get_filenames_sample(test_names, percentile)

    return sample_train_names, sample_test_names


def copy_files(source, destination, filenames, extension):
    print("Copying files from %s to %s" % (source, destination))
    files = os.listdir(source)
    percentage = 0
    for i, f in enumerate(files):
        if f.replace(extension, '') in filenames:
            src = os.path.join(source, f)
            dst = os.path.join(destination, f)

            copyfile(src, dst)
            if os.path.isfile(dst):
                print("Copied file %s to %s" % (src, dst))

                if percentage == 0:
                    print("%s/100" % str(percentage))
                if math.floor((i / len(files) * 100)) > percentage:
                    percentage = math.floor((i / len(files) * 100))
                    print("%s/100" % str(percentage))
            else:
                raise Exception("Could not copy file %s to %s" % (src, dst))



def create_sample(source, destination):
    p = 0.05
    sample_train_names, sample_test_names = get_sample_filenames(source, percentile=p)

    # save samples
    create_pickle(sample_train_names, os.path.join(destination, 'train'), 'filenames.pickle')
    create_pickle(sample_test_names, os.path.join(destination, 'test'), 'filenames.pickle')

    sample_filenames = sample_train_names + sample_test_names

    # Copy text files
    copy_files(source=os.path.join(source, 'text'),
               destination=os.path.join(destination, 'text'),
               extension='.txt',
               filenames=sample_filenames)

    # Copy images
    copy_files(source=os.path.join(source, 'train2014'),
               destination=os.path.join(destination, 'train2014'),
               extension='.jpg',
               filenames=sample_filenames)
    copy_files(source=os.path.join(source, 'val2014'),
               destination=os.path.join(destination, 'val2014'),
               extension='.jpg',
               filenames=sample_filenames)


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


def create_json(old_json_path, new_json_path, filenames):
    with open(old_json_path, 'rb') as old_json:
        data = json.load(old_json)
        new_json = copy.deepcopy(data)

    # Check if necessary to delete categories
    delete_categories = False
    if "categories" in new_json.keys():
        delete_categories = True

    img_len = len(new_json["images"])
    ann_len = len(new_json["annotations"])
    cat_len = len(new_json["categories"])

    # Filter images
    new_json, img_ids_to_del = delete_from_dict(obj_dict=new_json, obj_key="images", check_key="file_name",
                                                control_list=filenames, extension=".jpg", del_key="id")

    new_json, cat_ids_to_del = delete_from_dict(obj_dict=new_json, obj_key="annotations", check_key="image_id",
                                                control_list=img_ids_to_del, del_key="category_id")

    if delete_categories:
        new_json, del_list = delete_from_dict(obj_dict=new_json, obj_key="categories", check_key="id",
                                              control_list=cat_ids_to_del)


    print("Length of images before start: ", img_len)
    print("Length of annotations before start: ", ann_len)
    if delete_categories: print("Length of categories before start: ", cat_len)

    print("Length of images after start: ", len(new_json["images"]))
    print("Length of annotations after start: ", len(new_json["annotations"]))
    if delete_categories: print("Length of categories after start: ", len(new_json["categories"]))

    with open(new_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_json, outfile)

        print("Saved json to: ", new_json_path)


def check_json(file):
    with open(file, 'rb') as json_file:
        data = json.load(json_file)
        print(data["categories"][:10])
        for key, value in data.items():
            print("Key: ", key)


if __name__ == '__main__':

    big_root = '../data/big'
    small_root = '../data/small'


    train_filenames, test_filenames = get_sample_filenames(big_root, 0.05)

    print("About to copy files...")
    '''
    copy_files(source=os.path.join(big_root, 'train2014'), destination=os.path.join(small_root, 'train2014'),
               extension='.jpg', filenames=train_filenames)

    '''

    copy_files(source=os.path.join(big_root, 'val2014'), destination=os.path.join(small_root, 'val2014'),
               extension='.jpg', filenames=test_filenames)

    '''
    copy_files(source=os.path.join(big_root, 'train2014_resized'), destination=os.path.join(small_root, 'train2014_resized'),
               extension='.jpg', filenames=train_filenames)

    copy_files(source=os.path.join(big_root, 'val2014_resized'),
               destination=os.path.join(small_root, 'val2014_resized'),
               extension='.jpg', filenames=test_filenames)

    captions = load_pickle('../data/captions.pickle')
    for v in captions[0]:
        print(v)
    create_sample(root, small_root)

    check_json('../data/big/annotations/instances_train2014.json')

    create_json(os.path.join(root, 'annotations/captions_train2014.json'),
                os.path.join(small_root, 'annotations/captions_train2014.json'),
                train_filenames)
    create_json(os.path.join(root, 'annotations/captions_val2014.json'),
                os.path.join(small_root, 'annotations/captions_val2014.json'),
                test_filenames)


    create_json(os.path.join(big_root, 'annotations/instances_train2014.json'),
                os.path.join(small_root, 'annotations/instances_train2014.json'),
                train_filenames)
    create_json(os.path.join(big_root, 'annotations/instances_val2014.json'),
                os.path.join(small_root, 'annotations/instances_val2014.json'),
                test_filenames)
    create_json(os.path.join(big_root, 'annotations/person_keypoints_train2014.json'),
                os.path.join(small_root, 'annotations/person_keypoints_train2014.json'),
                train_filenames)

    create_json(os.path.join(big_root, 'annotations/person_keypoints_val2014.json'),
                os.path.join(small_root, 'annotations/person_keypoints_val2014.json'),
                test_filenames)
    '''





