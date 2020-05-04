# Text-To-Image Generator

Combining MirrorGAN and BERT for generating images based on natural language input.
Based on https://github.com/qiaott/MirrorGAN and https://github.com/ajamjoom/Image-Captions


Parameters to configure:
* Output dir
* model dir
* model paths

How-to:
FIRST CONFIGURE!

1. Train STEM by running pretrain_DAMSM.py
2. Train STREAM by running pretrain_STREAM.py
3. Move trained STEM and STREAM models from step 1 and 2 to model directory that is configured in config
4. Train the GAN by running main.py
5. To check results, configure Experients to point to where the trained GAN T2I generator and text encoder (from STEM) is saved and run... TO BE CONTINUED


