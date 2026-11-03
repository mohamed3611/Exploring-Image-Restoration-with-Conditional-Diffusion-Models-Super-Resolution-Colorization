# Image Restoration Using Conditional Diffusion Models

A Computer Vision project exploring two image restoration tasks using **Conditional Diffusion Models** and **PyTorch**:

- **Image Super-Resolution:** Reconstructing high-resolution images from low-resolution inputs.
- **Image Colorization:** Generating color images from grayscale inputs.

Both tasks use a conditional U-Net architecture with multi-scale feature extraction from a frozen, ImageNet-pretrained ResNet-50.

## Project Overview

This project explores how diffusion models can be conditioned on input images to perform image restoration.

The implementation includes:
1. Dataset preparation and preprocessing.
2. A diffusion noise schedule.
3. A conditional U-Net architecture.
4. Multi-scale feature extraction using ResNet-50.
5. Model training and checkpoint selection.
6. Image generation using DDIM sampling.
7. Quantitative evaluation using PSNR and SSIM.

## Tasks

### 1. Image Super-Resolution

The objective is to reconstruct a higher-resolution image from a low-resolution input.

**Dataset preparation:**
- Dataset: CIFAR-10.
- Original CIFAR-10 images are resized to 64 × 64.
- Images are downsampled to 16 × 16.
- The low-resolution images are then upsampled to 64 × 64 to create the model inputs.
- The corresponding 64 × 64 images serve as reconstruction targets.

**Dataset split:**

| Split | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |

**Evaluation Results:**

| Metric | Diffusion Model | Bicubic Baseline |
|---|---:|---:|
| PSNR | **27.5720** | 27.2039 |
| SSIM | **0.8513** | 0.8402 |

The diffusion model outperformed the bicubic baseline on both reported metrics.

### 2. Image Colorization

The objective is to generate an RGB image from a grayscale input.

**Dataset preparation:**
- Uses the same CIFAR-10 dataset and train/validation/test splits.
- RGB images are converted into grayscale inputs.
- The original RGB images are used as target outputs.

**Evaluation Results:**

| Metric | Diffusion Model | Baseline |
|---|---:|---:|
| PSNR | 22.3368 | **22.9884** |
| SSIM | 0.9064 | **0.9218** |

The colorization model did not outperform the baseline in this experiment. These results provide a useful reference point for further experimentation and model improvement.

## Model Architecture

The project uses a conditional diffusion framework with the following components:

- **Conditional U-Net:** Predicts the noise added to the target image.
- **ResNet-50 Feature Extractor:** Frozen ImageNet-pretrained backbone that extracts multi-scale image features.
- **Diffusion Schedule:** DDPM forward process with 1,000 timesteps.
- **DDIM Sampling:** Generates restored images using 50 inference steps.

The same general architecture is used for both tasks, with separate model training for super-resolution and colorization.

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Dataset | CIFAR-10 |
| Image Size | 64 × 64 |
| Low-Resolution Input | 16 × 16 |
| Batch Size | 64 on GPU |
| Epochs per Task | 40 |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Diffusion Timesteps | 1,000 |
| DDIM Sampling Steps | 50 |
| Learning Rate Scheduler | Cosine Annealing |
| Mixed Precision | Enabled on CUDA |

## Evaluation Metrics

**PSNR (Peak Signal-to-Noise Ratio)**

Measures the reconstruction quality by comparing the generated image with the target image. Higher values generally indicate lower reconstruction error.

**SSIM (Structural Similarity Index)**

Measures structural similarity between two images. Higher values indicate greater similarity.

## Technologies Used

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />
</p>

## Repository Structure

```text
Image-Restoration-Using-Conditional-Diffusion/
│
├── Week_10_Solved_Notebook.ipynb
└── README.md
```

The notebook contains the complete implementation, training process, evaluation, and visual results for both tasks.

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/mohamed3611/Image-Restoration-Using-Conditional-Diffusion.git
```

2. Open the notebook using Google Colab or Jupyter Notebook.

3. Run the cells sequentially.

4. The notebook downloads CIFAR-10 automatically and prepares the required datasets.

5. Train the super-resolution and colorization models, then inspect their generated outputs and evaluation metrics.

**Note:** Training is computationally intensive. A CUDA-enabled GPU is recommended.

## Important Experimental Notes

- CIFAR-10 images are originally 32 × 32 pixels. Resizing them to 64 × 64 creates a controlled synthetic restoration experiment, not genuine high-resolution image recovery.
- Quantitative evaluation uses a limited subset of the test data, so the reported metrics should be interpreted as experimental results rather than a comprehensive benchmark.
- The colorization results show that the model still has room for improvement compared with the baseline.

## Key Learning Outcomes

- Understanding conditional diffusion models.
- Implementing U-Net architectures in PyTorch.
- Using pretrained CNNs for multi-scale feature extraction.
- Preparing paired datasets for image restoration.
- Training deep learning models with AdamW and learning-rate scheduling.
- Applying DDIM sampling for image generation.
- Evaluating image reconstruction using PSNR and SSIM.

## Author

**Mohamed Tamer**

Computer Science Student | Aspiring AI Engineer

- **GitHub:** [mohamed3611](https://github.com/mohamed3611)
- **LinkedIn:** [Mohamed Tamer](https://www.linkedin.com/in/mohamed-tamer-27270837a/)
- **Email:** mohamedd.t.mamdouh@gmail.com

---

*This project was developed as part of my AI and Deep Learning coursework.*
