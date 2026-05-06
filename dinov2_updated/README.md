# Skin Lesion Classification Experiments

## Data Information

- The training dataset used in these scripts is **DermaMNIST**, sourced from [MedMNIST](https://medmnist.com/).
- The external validation dataset for dermoscopy skin lesion evaluation is sourced from [MedIMeta](https://www.woerner.eu/projects/medimeta/).

## Script Information

- Each script is self-contained and can be downloaded and run independently from end to end.
- It is recommended to use **HPC or GPU resources** for training **224×224** resolution models because of the higher computational cost.
- This is especially important for the **DINOv2** scripts. Detailed instructions are provided below.

## Folder Structure Overview

Experiments using **ResNet-50**, **ResNet-18**, **EfficientNet**, and **DINOv2** on both **224×224** and **28×28** resolution images are located in the main directory.  
The experiments specifically for **ResNet-18** and **ResNet-50** on **64x64**  are stored in the `medmnist_resnet_nick` folder.

The `eda` folder contains our initial exploratory data analysis (EDA) for **DermaMNIST** and other datasets that may be used later to evaluate **generalizability**.

## How to Run the Code

All scripts except those related to **DINOv2** can be run on **Google Colab**.  
For **DINOv2**, the RAM requirement is much higher, so Colab may sometimes crash. The following instructions describe how to run the DINOv2 experiments.

### Option A — Google Colab (Recommended)

No local setup is needed. Open  
`medmnist_dinov2_224_shupeng.ipynb` in Colab.

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
conda env create -f environment.yaml
conda activate dinov2-dermamnist
```

For CPU-only machines: edit environment.yaml before creating the environment and replace pytorch-cuda=12.1 with cpuonly.

2. Register the kernel with Jupyter
```
python -m ipykernel install --user --name dinov2-dermamnist
```
3. Launch the notebook
```
jupyter notebook medmnist_dinov2_224_shupeng.ipynb
```

If it is not selected automatically, choose the kernel dinov2-dermamnist.

4. First-run feature extraction

The DINOv2 weights (~330 MB) are downloaded from Hugging Face during the first run.
Pre-extracted features are cached under features/ and reused in later runs.

Approximate extraction time:
~10 minutes on a T4 GPU
~40 minutes on CPU

5. Outputs
The main output files and folders are listed below:

| File / Folder | Description |
|---------------|-------------|
| `features/cls/` | Cached DINOv2 CLS token features in `float32` format |
| `features/patch/` | Cached DINOv2 patch token features in `float16` format with shape `768×16×16` |
| `checkpoints/` | Best saved model weights for each classification head (`.pt`) |
| `224_training_curves.png` | Training loss and validation accuracy curves for all five heads |
| `224_method_acc_auc.png` | Comparison of method-level Accuracy, Macro AUC, and Weighted AUC |
| `224_perclass_auc_heatmap.png` | Heatmap of per-class AUC scores |
| `224_perclass_recall_heatmap.png` | Heatmap of per-class Recall scores |
| `224_perclass_grouped.png` | Grouped bar chart showing per-class AUC and Recall |
