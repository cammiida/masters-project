import os
import random
import torchvision.utils as vutils
from tqdm import tqdm
import torchvision.transforms as transforms

from experiments.experimenter import Experimenter
from datasets import TextDataset
from miscc.utils import mkdir_p
from cfg.config import cfg as cfg


def save_images(save_dir, images):
    if not os.path.isdir(save_dir):
        mkdir_p(save_dir)

    for image in images:
        img_num = len(os.listdir(save_dir))
        img_path = os.path.join(save_dir, 'img_%d.jpg' % img_num)
        vutils.save_image(image, img_path, normalize=True)


    print('Saved img batch to %s' % save_dir)

def delete_file(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
        print('%s deleted' % filepath)


def get_dataset_props(data_dir):
    big_data_dir = os.path.join(cfg.DATA_DIR, 'big')
    captions_path = os.path.join(cfg.DATA_DIR, cfg.DATA_SIZE, 'captions.pickle')
    delete_file(captions_path)

    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = TextDataset(big_data_dir, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    n_words = dataset.n_words
    wordtoix = dataset.wordtoix

    delete_file(captions_path)

    dataset = TextDataset(data_dir, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    captions = []
    for filename in dataset.filenames:
        filepath = '%s/text/%s.txt' % (data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            caps = f.read().split('\n')
            cap = ''
            trials = 0
            # Take a random caption
            while cap == '' or trials > cfg.TEXT.CAPTIONS_PER_IMAGE:
                rand = random.randint(0, cfg.TEXT.CAPTIONS_PER_IMAGE - 1)
                if caps[rand] != '': cap = caps[rand]
                trials += 1
            captions.append(cap)

    delete_file(captions_path)

    counter = 0
    for cap in captions:
        if cap == '':
            counter += 1

    print("num empty captions: ", counter)
    print("len captions: ", len(captions))
    print("len filenames: ", len(dataset.filenames))

    return n_words, wordtoix, captions


def gen_and_save_imgs(output_dir):
    data_dir = os.path.join(cfg.DATA_DIR, cfg.DATA_SIZE)
    n_words, wordtoix, captions = get_dataset_props(data_dir)

    batch_size = cfg.TRAIN.BATCH_SIZE
    cap_batches = [captions[x:x+batch_size] for x in range(0, len(captions), batch_size)]
    print("captions: ", captions)
    print(len(cap_batches))

    experimenter = Experimenter(embedding_dim=cfg.TEXT.EMBEDDING_DIM, net_E=cfg.TRAIN.NET_E,
                                n_words=n_words, wordtoix=wordtoix)
    for cap_batch in tqdm(cap_batches):
        images = experimenter.generate_images(cfg.TRAIN.NET_G, z_dim=cfg.GAN.Z_DIM, sentences=cap_batch)[-1]
        #print(images)
        save_images(output_dir, images)
