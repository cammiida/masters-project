from experiments.pytorch_fid.gen_and_save_imgs import gen_and_save_imgs
from experiments.pytorch_fid.fid_score import calculate_fid_given_paths

import os, shutil
import argparse
import torch
import pprint

from cfg.config import cfg, cfg_from_file

def delete_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images from validation set')
    parser.add_argument('--ref_imgs_dir', dest='ref_imgs_dir', type=str, default='val2014_resized')
    parser.add_argument('--cfg', dest='cfg_file', default='cfg/experiments/fid_original.yml', help='optional config file', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print('Using config:')
    pprint.pprint(cfg)

    print('Generating and saving images from validation set...')
    output_dir = os.path.join(cfg.OUTPUT_PATH, cfg.CONFIG_NAME)
    gen_and_save_imgs(output_dir)
    print('Generated and saved images.')

    print('Calculating FID score...')
    ref_imgs_dir = os.path.join(cfg.DATA_DIR, cfg.DATA_SIZE, args.ref_imgs_dir)
    cuda = True if cfg.CUDA and torch.cuda.is_available() else False
    fid_score = calculate_fid_given_paths([ref_imgs_dir, output_dir], batch_size=cfg.TRAIN.BATCH_SIZE, cuda=cuda, dims=2048)
    print('The FID score is: ', fid_score)

    print('Deleting generated images...')
    delete_files(output_dir)

