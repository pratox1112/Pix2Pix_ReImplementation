# Pix2Pix with VGG19 Perceptual Loss (PyTorch Re-Implementation)

This repository provides a clean, modern re-implementation of **Pix2Pix** using PyTorch, extended with a **VGG19 perceptual loss** to significantly improve texture quality, sharpness, and structural details in the generated images.

This re-implementation is based on the official  
[`pytorch-CycleGAN-and-pix2pix`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
repository, with modifications to the **generator loss** and **network architecture** to incorporate a perceptual feature matching term.

---

## 🚀 Key Features
- Full PyTorch implementation of **Pix2Pix**
- **VGG19 Perceptual Loss** integrated into the generator objective
- Improved edge clarity, texture detail, and structure preservation
- Fully compatible with GPU environments (CUDA 12.1)
- Works with PyTorch 2.4.0 and Python 3.11
- Includes training, testing, visualization, and HTML output pages
- Clean and minimal environment setup using `environment.yml`

---

## 🔧 What’s New in This Version? (Your Contribution)

### ✔ VGG19 Perceptual Loss Added
A pretrained VGG19 model (ImageNet) is used to extract mid-level feature maps  
(layer `conv3_3`, index 16). The perceptual loss encourages:

- Sharper edges  
- Stronger texture reconstruction  
- More realistic structure  
- Reduced blurring compared to standard L1 loss  

The new generator objective becomes:

\[
\mathcal{L}_G =
\mathcal{L}_{GAN}
+ \lambda_{L1} \mathcal{L}_{L1}
+ \lambda_{perc} \mathcal{L}_{perc}
\]

### Files Modified:
- `models/pix2pix_model.py`  
- `models/networks.py` (added VGG19 feature extractor)

---

## 📁 Folder Structure

```
pytorch-CycleGAN-and-pix2pix/
│
├── train.py
├── test.py
│
├── models/
│   ├── pix2pix_model.py          # Modified to include perceptual loss
│   ├── networks.py               # Modified VGG19 extractor added
│   ├── base_model.py
│
├── data/
│   ├── aligned_dataset.py        # Paired A|B loader
│   ├── base_dataset.py
│
├── options/
│   ├── base_options.py
│   ├── train_options.py
│   ├── test_options.py
│
├── util/
│   ├── html.py                   # Training/testing HTML pages
│   ├── util.py
│   ├── visualizer.py
│
└── datasets/
    └── facades/
        ├── train/
        ├── test/
```

---

## 🛠️ Environment Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

Example `environment.yml`:

```yaml
name: pytorch-img2img
channels:
  - pytorch
  - conda-forge
  - nvidia
dependencies:
  - python=3.11
  - pytorch=2.4.0
  - torchvision=0.19.0
  - pytorch-cuda=12.1
  - numpy=1.24.3
  - scikit-image
  - pip
  - pip:
      - dominate>=2.8.0
      - Pillow>=10.0.0
      - wandb>=0.16.0
```

---

## 📥 Download Dataset

```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```

Or manually place data in:

```
datasets/facades/train/
datasets/facades/test/
```

---

## 🏋️ Training (with Perceptual Loss)

```bash
python train.py   --dataroot ./datasets/facades   --name facades_pix2pix_vgg   --model pix2pix   --dataset_mode aligned   --direction BtoA   --lambda_perceptual 10
```

Training results appear in:

```
checkpoints/facades_pix2pix_vgg/web/index.html
```

---

## 🔍 Testing

```bash
python test.py   --dataroot ./datasets/facades   --name facades_pix2pix_vgg   --model pix2pix   --dataset_mode aligned   --direction BtoA
```

Results saved to:

```
results/facades_pix2pix_vgg/test_latest/index.html
```

---

## 📊 Expected Results

Adding perceptual loss improves:

- Texture detail  
- Structural integrity  
- Sharpness  
- Overall perceptual quality  

---

## 📜 Citation

If you use this code, cite:

**Original Pix2Pix paper:**  
_Isola et al., “Image-to-Image Translation with Conditional Adversarial Networks”, CVPR 2017._

**Original Repo:**  
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
