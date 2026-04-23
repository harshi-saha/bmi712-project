
**Data Information:**
- The training dataset used in these scripts is DermaMNIST sourced from [MedMNIST](https://medmnist.com/)
- The external validation dataset that is dermoscopy skin lesion evaluation sourced from [MedIMeta](https://www.woerner.eu/projects/medimeta/)

**Scipt Information:**
- Each of these scripts is self contained as can be downloaded and run end to end independently
- It is recommended to use HPC or GPU resources to run 224x224 resolution models due to the required computational resources
- It is also especially recommended to do so for the DinoV2 scripts and some directions are provided below

## Folder Structure Overview

Experiments using **ResNet-50**, **EfficientNet**, and **DINOv2** on both **224×224** and **28×28** resolution images are located in the main directory.  
The experiments specifically for **ResNet-50** are stored in the `medmnist_resnet_nick` folder.  

The `eda` folder contains our initial exploratory data analysis (EDA) for **DermaMNIST** and other datasets that may be used later to evaluate **generalizability**.

## How to Run the Code

All scripts except those related to **DINOv2** can be run on **Google Colab**.  
For DINOv2, the RAM requirement is much higher, so Colab may sometimes crash. The following instructions describe how to run the DINOv2 experiments.

### Option A — Google Colab (Recommended)

No local setup is needed. Open  
`code/medmnist_dinov2_224_shupeng.ipynb` in Colab.

1. Go to **Colab** → **File → Upload notebook** → select the `.ipynb` file.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**.  
   Note: extra RAM may also be required.
3. Run all cells with **Runtime → Run all**.
   - Cell 1 installs all required dependencies automatically.
   - The **DermaMNIST 224×224** dataset is downloaded during the first run.
   - Extracted features are cached in `/content/features/`, so later runs can skip feature extraction.
4. After training finishes, **Cell 8** packages all plots and checkpoints into `dermamnist_224_results.zip` and downloads it automatically.

---

### Option B — Local or Cluster Setup

#### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- A CUDA-capable GPU (optional; CPU also works, but training will be slower)

#### 1. Create the conda environment

```bash
conda env create -f code/environment.yaml
conda activate dinov2-dermamnist
