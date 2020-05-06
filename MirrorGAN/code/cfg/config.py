from __future__ import division, print_function
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset name: coco
__C.DATASET_NAME = 'coco'
__C.CONFIG_NAME = 'MirrorGAN'
__C.ROOT_DATA_DIR = '../../data'
__C.DATASET_SIZE = 'small'
__C.DATA_DIR = ''
__C.MODELS_DIR = '../../models'

__C.DEVICE = 'cpu'
__C.WORKERS = 8
__C.OUTPUT_PATH = '../..'
__C.RNN_TYPE = 'LSTM'
__C.B_VALIDATION = False

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64


# TRAINING OPTIONS
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32 # Should be 64, but takes forever to get resources...
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.DECODER_LR = 4e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True

__C.TRAIN.NET_E = 'STEM/text_encoder.pth'
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 0.0
__C.TRAIN.SMOOTH.LAMBDA1 = 1.0




# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
# TODO: Make it DCGAN?
__C.GAN.B_DCGAN = False


__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 5
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18


# Vocab indices
__C.VOCAB = edict()
__C.VOCAB.PAD = 0
__C.VOCAB.START = 1
__C.VOCAB.END = 2
__C.VOCAB.UNK = 3
__C.VOCAB.THRESHOLD = 4

# CAPTION/STREAM MODEL SETTINGS
__C.STREAM = edict()
__C.STREAM.USE_ORIGINAL = False
__C.STREAM.EMBED_SIZE = 256
__C.STREAM.HIDDEN_SIZE = 512
__C.STREAM.NUM_LAYERS = 1
__C.STREAM.LEARNING_RATE = 0.001
__C.STREAM.CAPTION_CNN_PATH = 'STREAM/cnn_encoder'
__C.STREAM.CAPTION_RNN_PATH = 'STREAM/rnn_decoder'
__C.STREAM.PREPROCESS_DATA = True

__C.STREAM.GRAD_CLIP = 5.
__C.STREAM.FROM_CHECKPOINT = False
__C.STREAM.TRAIN_MODEL = True
__C.STREAM.VALID_MODEL = False


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)