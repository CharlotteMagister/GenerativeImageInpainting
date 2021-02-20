# Generative Image Inpainting

This repository contains the code for the paper "Generative Image Inpainting for Retinal Images using Generative Adversarial Networks".

The 'src/' directory contains the source code for the project. It includes all GAN architectures, the module for inpainting and the module for evaluation. Moreover, it contains the subdirectory 'stitching/', which contains the code written for image stitching. As mentioned in the report, the focus resided on the task of inpainting wherefore the code is separated out. 

Any output produced by the code will be saved to the 'output/' directory. The directory will contain sub-directories named after the models, which in turn will contain graphs and sample images.

* Please note this directory only contains source code and no data, wherefore, the code will not run without any prior set up.

## Instructions for Preprocessing Step

1. In the command line, navigate to the 'src/' directory.
2. Create a python virtual environments by following the following instructions:
    1. To create the environment run: python3 -m venv [envname]
    2. To activate the environment run: source [envname]/bin/activate
    3. Import the required modules by running: pip3 install -r requirements.txt
3. To perform the preprocessing run: python3 split_images.py
4. Navigate to the 'input/' folder. The 'newResults/' folder contains the results.

## Instructions for Training a GAN

1. In the command line, navigate to the 'src/' directory.
2. (If done can omitted.) Create a python virtual environments by following the following instructions:
    1. To create the environment run: python3 -m venv [envname]
    2. To activate the environment run: source [envname]/bin/activate
    3. Import the required modules by running: pip3 install -r requirements.txt
3. Test input is provided for training.
4. Any GAN architecture can now be run by running the corresponding file:
    1. VanillaGAN: python3 VanillaGAN.py
    2. DCGAN: python3 DCGAN.py
    3. WGAN: python3 WGAN.py
    4. WGAN 128x128px: python3 large_WGAN.py
    5. Unrolled GAN: UnrolledGAN.py
5. The output will be saved into the 'newResults/' folder in the folder of the respective GAN in the 'output/' folder to avoid overwritting the data submitted.

## Instructions for Image Inpainting

* Please note that the inpainting module uses the WGAN 64x64px model trained, as it is the final
model chosen.

1. In the command line, navigate to the 'src/' directory.
2. (If done can omitted.) Create a python virtual environments by following the following instructions:
    1. To create the environment run: python3 -m venv [envname]
    2. To activate the environment run: source [envname]/bin/activate
    3. Import the required modules by running: pip3 install -r requirements.txt
3. Run the module providing an image to be inpainted:
    * python3 inpaint.py [path_to_img]
    * e.g. python3 inpaint.py ../input/evaluationData/data/visit_1_image_4.png
    * A warning will be outputted that the source code has changed, which is due to comments
    and renaming of variables being done after the model was trained.
4. The output can be found under the path 'output/inpainting/newResults'.

## Instructions for Evaluating the Output

* Please note that the evaluation module uses the WGAN 64x64px model trained, as it is the final
model chosen.

1. In the command line, navigate to the 'src/' directory.
2. (If done can omitted.) Create a python virtual environments by following the following instructions:
    1. To create the environment run: python3 -m venv [envname]
    2. To activate the environment run: source [envname]/bin/activate
    3. Import the required modules by running: pip3 install -r requirements.txt
3. Run the evaluation by running: python3 evaluate.py
4. The results are outputted directly to the terminal.
