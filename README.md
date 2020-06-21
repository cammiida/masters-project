# Text-To-Image Generator

Combining MirrorGAN and BERT for generating images based on natural language input.
Based on https://github.com/qiaott/MirrorGAN and https://github.com/ajamjoom/Image-Captions.

# Instructions to run the code

TODO: Mention configuration of paths in config
Install from requirements.txt with ``$ pip install requirements.txt``.


## Download and clean data
1. Create three folders: (1) data, (2) models, and (3) output.
2. Within folder each folder, create folders named 'big' and 'small'. See the main folder structure at the bottom of this page if you are unsure.
3. Within data > big, create a folder named 'annotations'.
4. Download the train 2014 and val2014 from the MS COCO dataset (http://cocodataset.org/#download) and place them inside the data folder.
5. Download the COCO train/val 2014 captions (also from http://cocodataset.org/#download) and place them inside the data > big > annotations folder.
6. Run the process_data.py file. This will generate the train2014_resized, val2014_resized, and vocab.pkl inside the data > big folder.

## Train STEM

## Train STREAM

## Train GAN
Make sure the pretrained STEM and STREAM models are placed in the models folder.

Parameters to configure:
* Output dir
* model dir
* model paths


1. Train STEM by running pretrain_DAMSM.py
2. Train STREAM by running pretrain_STREAM.py
3. Move trained STEM and STREAM models from step 1 and 2 to model directory that is configured in config
4. Train the GAN by running main.py
5. To check results, configure Experiments to point to where the trained GAN T2I generator and text encoder (from STEM) is saved and run... TO BE CONTINUED


## Pre-Trained Models
1. STEM
2. STREAM
3. Generator and Discriminators




# Main folder structure:
```
project
│   .gitignore
│   README.md
│   requirements.txt
└───code
│   │   ...
│   
└───data
│   └---big
│   │   │   example_captions.txt
│   │   │   example_filenames.txt
│   │   │   interpolate_captions.txt
│   │   │   vocab.pkl
│   │   └---annotations
│   │   │
│   │   └---test
│   │   │
│   │   └---text 
│   │   │
│   │   └---train
│   │
│   └---small
│   │   │   ... (same as in big/)
│
└---models
│   │   └---big
│   │   │   └---GAN
│   │   │   │
│   │   │   └---STEM
│   │   │   │
│   │   │   └---STREAM
│   │   │
│   │   └---small
│   │   │   ... (same as in big/)   
│
└---output
│   │   └---big
│   │   │
│   │   └---small

   
```