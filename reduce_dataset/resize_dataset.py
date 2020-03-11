import os
import pickle
import random
import math
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


def create_pickle(list, dir, filename):
    # TODO: Check that file doesn't exist
    print("%s is directory: %s" % (dir, os.path.isdir(dir)))
    if not os.path.isdir(dir):
        create_dir(dir)
    with open(os.path.join(dir, filename), 'wb') as f:
        pickle.dump(list, f)



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


def copy_files(filenames, source, destination, extension):
    files = os.listdir(source)
    for f in files:
        if f.replace(extension, '') in filenames:
            src = os.path.join(source, f)
            dst = os.path.join(destination, f)

            try:
                copyfile(src, dst)
            except OSError:
                print("Could not copy file %s to %s" % (src, dst))
            else:
                print("Copied file %s to %s" % (src, dst))




def create_sample(data_dir, small_data_dir):
    p = 0.05
    sample_train_names, sample_test_names = get_sample_filenames(data_dir, percentile=p)

    # save samples
    create_pickle(sample_train_names, os.path.join(small_data_dir, 'train'), 'filenames.pickle')
    create_pickle(sample_test_names, os.path.join(small_data_dir, 'test'), 'filenames.pickle')

    sample_filenames = sample_train_names + sample_test_names

    # Copy text files
    copy_files(sample_filenames,
               from_path=os.path.join(data_dir, 'text'),
               to_path=os.path.join(small_data_dir, 'text'),
               extension='.txt')

    # Copy images
    copy_files(sample_filenames,
               from_path=os.path.join(data_dir, 'train2014'),
               to_path=os.path.join(small_data_dir, 'train2014'),
               extension='.jpg')
    copy_files(sample_filenames,
               from_path=os.path.join(data_dir, 'val2014'),
               to_path=os.path.join(small_data_dir, 'val2014'),
               extension='.jpg')



if __name__ == '__main__':

    root = '../data'
    small_root = os.path.join(root, 'small')

    captions = load_pickle('../data/captions.pickle')
    for v in captions[0]:
        print(v)
    #create_sample(root, small_root)





