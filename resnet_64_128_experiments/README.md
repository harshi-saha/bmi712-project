# ResNet 64 and 128 Experiments

## Description

This directory contains various experiments performed using images of size 64x64 and 128x128.

## Files

The files `resnet18_experiments.ipynb` and `resnet50_experiments.ipynb` contain some initial experiments
on the 64x64 images using the corresponding ResNet models.

`resnet18_experiments.ipynb` contains:
* basic model fitting
* training for many epochs to evaluate when the models overfit
* testing various data augmentations to see how they affect performance
* a minor attention test using SE attention

`resnet50_experiments.ipynb` contains:
* some very basic EDA (but minimal since much of the EDA was already done in other notebooks)
* basic model fitting
* training for many epochs to evaluate when the models overfit
* testing using augmented data to see how it affects performance

The file `medmnist_resnet_nick.ipynb` contains further experiments using ResNet18 and ResNet50, this
time on both 64x64 images and 128x128 images. This file includes:
* basic EDA of the DermaMNIST dataset
* model training with simplified data augmentation, and with sweeps over the following settings:
    * image size: 64x64 or 128x128
    * class-weighted vs. unweighted loss function
    * ResNet 18 vs. 50
    * Attention (CBAM) vs. no attention
    * Learning rate 1e-3 vs. 1e-4
* visualization of results using GradCAM

The folder `helpers` contains files with many helper functions used by the Jupyter notebooks. These
helper functions were placed into these files to save space within the notebooks and ensure the notebooks
can be dedicated to experimentation. In order for the notebooks to run, the code must have the following
structure:

resnet_64_128_experiments/
├── helpers
│   ├── __init__.py
│   ├── attention.py
│   ├── constants.py
│   ├── data.py
│   ├── device.py
│   ├── gradcam.py
│   ├── metrics.py
│   ├── resnet.py
│   └── train.py
├── medmnist_resnet_nick.ipynb
├── resnet18_experiments.ipynb
└── resnet50_experiments.ipynb

## Reproducing Analysis

It is recommended that you run the Jupyter notebooks using a Google Colab runtime on the latest version
(2026.04 as of this writing). The Jupyter notebooks have commands to install a few packages, such as 
the `medmnist` and `medimeta` packages, but aside from those they assume the standard packages in a 
Google Colab runtime are present.

Before running the Jupyter notebooks on Google Colab, please ensure the following:

First, please make sure the `helpers` folder is present on the Colab file system. Your file structure
on Colab should look like what is shown above.

Second, for the notebook `medmnist_resnet_nick.ipynb`, you will need to have the MedIMeta dataset 
saved in Google Drive. It can be downloaded from here: 
https://www.woerner.eu/projects/medimeta/.

The MedIMeta dataset is a .zip file whose contents should look like this:

MedIMeta/
└── derm
    ├── LICENSE
    ├── annotations.csv
    ├── images  [11720 images inside]
    ├── images.hdf5
    ├── info.yaml
    ├── original_splits
    │   ├── test.txt
    │   ├── train.txt
    │   └── val.txt
    ├── splits
    │   ├── test.txt
    │   ├── train.txt
    │   └── val.txt
    ├── task_labels
    │   └── disease category.npy
    └── teaser.png

You will need to unzip the file and place it in your Google Drive. You will also need to set the
path `DRIVE_PROJECT_PATH` in the file `medmnist_resnet_nick.ipynb` to the path to the directory
containing the `MedIMeta` folder in your Google Drive. For example, if `medmnist_resnet_nick.ipynb` 
is located in `/content` and the `MedIMeta` folder is located in `/content/drive/MyDrive/School/Harvard/bmi712/project`
then you must set `DRIVE_PROJECT_PATH = 'drive/MyDrive/School/Harvard/bmi712/project'`

Once all these are set up, you should be able to run any of the Jupyter notebooks end-to-end to reproduce
the analysis.
