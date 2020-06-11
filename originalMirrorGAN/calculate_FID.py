from experiments.pytorch_fid.gen_and_save_imgs import gen_and_save_imgs
from experiments.pytorch_fid.fid_score import calculate_fid_given_paths

import os, shutil
import argparse
import torch

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
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='../output/experiments/FID_images')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='../data/')
    parser.add_argument('--models_dir', dest='models_dir', type=str, default='../models')
    parser.add_argument('--data_size', dest='data_size', type=str, default='big')
    parser.add_argument('--model_version', dest='model_version', type=str, default='new')
    parser.add_argument('--epoch', dest='epoch', type=int, default=50)
    parser.add_argument('--ref_imgs_dir', dest='ref_imgs_dir', type=str, default='val2014_resized')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print('Generating and saving images from validation set...')
    output_dir = os.path.join(args.output_dir, args.model_version)
    gen_and_save_imgs(output_dir, args.batch_size, args.model_version, args.epoch, args.data_dir, args.models_dir, args.data_size)
    print('Generated and saved images.')


    print('Calculating FID score...')
    ref_imgs_dir = os.path.join(args.data_dir, args.data_size, args.ref_imgs_dir)
    cuda = False
    if torch.cuda.is_available():
        cuda = True
    fid_score = calculate_fid_given_paths([ref_imgs_dir, output_dir], batch_size=args.batch_size, cuda=cuda, dims=2048)
    print('The FID score is: ', fid_score)

    print('Deleting generated images...')
    delete_files(args.output_dir)

