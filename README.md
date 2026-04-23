
**Data Information:**

**Scipt Information:**

**Notes for running DinoV2 scripts:**

Option A: Google Colab (Recommended)

No local setup needed. Open
`code/medmnsit_dinov2_224_shupeng.ipynb` in Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com) →
   **File → Upload notebook** → select the `.ipynb` file.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells (**Runtime → Run all**).
   - Cell 1 installs all dependencies automatically.
   - DermaMNIST 224×224 is downloaded on first run and features are
     cached to `/content/features/`; subsequent runs skip extraction.
4. After training completes, Cell 8 packages all plots and checkpoints
   into `dermamnist_224_results.zip` and downloads it automatically.

---

Option B: Local Setup

Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- CUDA-capable GPU (optional; CPU works but training is slower)

1. Create the conda environment

```bash
conda env create -f code/environment.yaml
conda activate dinov2-dermamnist
```

> **CPU-only machines:** edit `environment.yaml` before creating —
> replace the `pytorch-cuda=12.1` line with `cpuonly`.

2. Register the kernel with Jupyter

```bash
python -m ipykernel install --user --name dinov2-dermamnist
```

3. Launch the notebook

```bash
jupyter notebook code/medmnsit_dinov2_224_shupeng.ipynb
```

Select kernel **dinov2-dermamnist** if not chosen automatically.

4. First-run feature extraction

DINOv2 weights (~330 MB) are downloaded from HuggingFace on first run.
Pre-extracted features are cached under `features/` and reused on
subsequent runs. Extraction time: ~10 min on T4 GPU, ~40 min on CPU.

5. Outputs

| File | Description |
|------|-------------|
| `features/cls/` | DINOv2 CLS token features (float32) |
| `features/patch/` | DINOv2 patch token features (float16, 768×16×16) |
| `checkpoints/` | Best model weights for each head (`.pt`) |
| `224_training_curves.png` | Loss & val-accuracy curves (all 5 heads) |
| `224_method_acc_auc.png` | Method-level Accuracy / Macro AUC / Weighted AUC |
| `224_perclass_auc_heatmap.png` | Per-class AUC heatmap |
| `224_perclass_recall_heatmap.png` | Per-class Recall heatmap |
| `224_perclass_grouped.png` | Grouped bar: AUC + Recall by class |
