# Text-To-Image Generator

Combining MirrorGAN and BERT for generating images based on natural language input.
Based on [https://github.com/qiaott/MirrorGAN][MirrorGAN] and [https://github.com/ajamjoom/Image-Captions][github_img_caps].

## Instructions to run the code

Install from requirements.txt with ``$ pip install requirements.txt``.


### Download and clean data
1. Create three folders: (1) data, (2) models, and (3) output.
2. Within folder each folder, create folders named 'big' and 'small'. See the main folder structure at the bottom of this page if you are unsure.
3. Within data > big, create a folder named 'annotations'.
4. Download the train 2014 and val2014 from the [MS COCO dataset][MS_COCO] and place them inside the data folder.
5. Download the COCO train/val 2014 captions (also from the [MS COCO dataset][MS_COCO]) and place them inside the data > big > annotations folder.
6. (Optional) Run the ``resize_dataset.py`` file in the reduce_dataset/ folder to create a smaller version of the dataset. This will be added to the 'small' folder inside the 'data' folder.
7. Make sure all base configuration and hyper-parameter values are correct in ``cfg/config.py``.

### Train STEM
1. Make the sure configuration and hyper-parameters are correctly set in cfg/pretrain_STEM.yml.
2. Run ``pretrain_STEM.py`` with ``cfg/pretrain_STEM.yml`` supplied as an argument.

### Train STREAM
1. Make the sure configuration and hyper-parameters are correctly set in ``cfg/pretrain_STREAM.yml`` and ``cfg/pretrain_STREAM_original.yml``.
2. Train the new version: Run ``pretrain_STREAM.yml`` with ``cfg/pretrain_STREAM.yml`` supplied as an argument.
3. Train the original version: Run ``pretrain_STREAM.yml`` with ``cfg/pretrain_STREAM_original.yml`` supplied as an argument.


### Train GAN
1. Make the sure configuration and hyper-parameters are correctly set in ``cfg/train_coco.yml`` and ``cfg/train_coco_original.yml``. Make sure to specify the correct path to the pretrained STEM and STREAM models.
2. Train the new version: Run ``main.py`` with ``cfg/train_coco.yml`` supplied as an argument. Make sure to specify the path to the pre-trained new version of the STREAM models.
3. Train the original version: Run ``main.py`` with ``cfg/train_coco_original.yml`` supplied as an argument. Make sure to specify the path to the pre-trained original version of the STREAM models.

### Run experiments
- To generate images using the trained models, run ``generate_images.py`` with ``cfg/experiments/gen_imgs_new.py`` or 
``cfg/experiments/gen_imgs_original.yml`` supplied as an arugment.
- To calculate the FID score for the new or orginal model, run  ``calculate_FID.py`` with either ``cfg/experiments/fid_new.yml`` or ``cfg/experiments/fid_original.yml``.
- To calculate the BLEU score of the two STREAM modules, run ``pretrain_STREAM.yml`` supplied with ``cfg/validate_STREAM.yml`` and ``cfg/validate_STREAM_original.yml`` 
to calculate the score for the new and original versions, respectively.


## Main folder structure:
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

[MS_COCO]: http://cocodataset.org/#download
[github_img_caps]: https://github.com/ajamjoom/Image-Captions
[MirrorGAN]: https://github.com/qiaott/MirrorGAN