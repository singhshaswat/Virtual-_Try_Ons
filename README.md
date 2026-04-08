# DA-VITON — Depth-Aware Virtual Try-On Network

> **A GAN + Multi-Head Attention framework for high-resolution image-based virtual try-on on the VITON-HD (Zalando) dataset.**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Innovations](#key-innovations)
3. [Dataset](#dataset)
4. [Architecture](#architecture)
5. [Loss Functions](#loss-functions)
6. [Training Configuration](#training-configuration)
7. [Evaluation Metrics & Results](#evaluation-metrics--results)


---

## Overview

DA-VITON (**D**epth-**A**ware **VITON**) is a virtual try-on system that transfers a target garment onto a clothed person image. Unlike prior GAN-based methods, DA-VITON integrates:

- **Depth maps** for 3D spatial awareness and occlusion handling
- A **Garment Refinement Module (GRM)** that eliminates interior collar/sleeve regions from garment masks using monocular depth estimation
- **Multi-Head Cross-Attention** to align fine-grained garment texture and patterns with the person's body structure

The model is trained and evaluated on the **VITON-HD / Zalando dataset** at resolution **512×384** (extendable to 1024×768).

---

## Key Innovations

| Innovation | Description |
|---|---|
| Garment Refinement Module (GRM) | Depth-guided binary mask refinement to remove inner collar/sleeve regions before warping |
| Depth-Augmented Body Encoder | Combines agnostic image + DensePose + depth + OpenPose for rich spatial body representation |
| Multi-Head Cross-Attention | Garment encoder features act as key/value; body features act as query — enabling precise texture injection |
| Spectral-Norm U-Net Generator | High-resolution try-on synthesis with skip connections and instance normalisation |
| Conditional PatchGAN Discriminator | Realism feedback conditioned on the agnostic person image |

---

## Dataset

**VITON-HD (High-Resolution Zalando Dataset)**

| Split | Pairs |
|---|---|
| Train | 11,647 |
| Test | 2,032 |
| **Total** | **13,679** |

Each sample contains the following modalities:

| Modality | Shape | Description |
|---|---|---|
| `image` | `(3, 512, 384)` | Ground-truth person photo |
| `agnostic` | `(3, 512, 384)` | Person image with clothing region masked |
| `densepose` | `(3, 512, 384)` | Dense body UV map |
| `pose_img` | `(3, 512, 384)` | OpenPose skeleton render |
| `parse` | `(512, 384)` int64 | 20-class body-part segmentation |
| `cloth` | `(3, 512, 384)` | Flat-lay target garment |
| `cloth_mask` | `(1, 512, 384)` | Binary garment mask |

Dataset path: `/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset`

---

## Architecture

The DA-VITON pipeline consists of four sequential modules:

### 1. Garment Refinement Module (GRM)
- Lightweight U-Net-style **DepthNet** (3 encoder + 3 decoder blocks)
- Estimates monocular depth of the flat-lay garment
- Applies a depth threshold (`0.35`) to identify interior regions (collar lining, sleeve inner)
- Produces a **refined binary mask** that removes artifact-prone interior areas

| Property | Value |
|---|---|
| Parameters | 190,481 (190.5K) |
| Depth threshold | 0.35 |
| Output tensors | `refined_mask (B,1,512,384)`, `refined_cloth (B,3,512,384)`, `depth_map (B,1,512,384)` |
| Typical mask reduction | ~5.7% interior pixels removed |
| Inference speed | ~2.8 ms / batch of 2 |

### 2. Garment Encoder
- ResNet-style backbone with 4 downsampling stages (MaxPool)
- Self-Attention (Multi-Head, 8 heads) at the bottleneck feature map
- Input: garment RGB (3ch) + refined mask (1ch) = **4 channels**

| Feature Map | Shape |
|---|---|
| `f1` | `(B, 64, 128, 96)` |
| `f2` | `(B, 128, 64, 48)` |
| `f3` | `(B, 256, 32, 24)` |
| `f4` (+self-attn) | `(B, 512, 32, 24)` |
| **Total parameters** | **12,231,296** |

### 3. Body Encoder
- Same ResNet-style architecture with Self-Attention at bottleneck
- Input: agnostic (3ch) + DensePose (3ch) + depth (1ch) + OpenPose (3ch) = **10 channels**
- Outputs multi-scale body features + depth map

| **Total parameters** | **12,440,593** |
|---|---|

### 4. Garment–Body Integration Module + Try-On Generator
- **Cross-Attention** layer: body features (query) × garment features (key/value) → aligned representation
- **U-Net decoder** with skip connections at 3 scales
- Produces a 3-channel garment representation + 20-class segmentation logits
- **TryOnGenerator**: Spectral-Norm U-Net with 6 encoder / 6 decoder blocks + InstanceNorm + skip connections
- Outputs the final try-on image via `tanh` activation

| Module | Parameters |
|---|---|
| GarmentBodyIntegration | 3,008,183 |
| TryOnGenerator | 21,418,691 |

### Full Model Summary

| Module | Parameters | Description |
|---|---|---|
| GarmentRefinementModule | 190,481 | Depth-guided mask refinement |
| GarmentEncoder | 12,231,296 | ResNet + Self-Attention |
| BodyEncoder | 12,440,593 | ResNet + Depth + Self-Attention |
| GarmentBodyIntegration | 3,008,183 | Cross-Attention + U-Net decoder |
| TryOnGenerator | 21,418,691 | Spectral-Norm U-Net (HR synthesis) |
| **Total** | **~49.3M** | End-to-end GAN model |

---

## Loss Functions

The generator is trained with a weighted multi-term objective (paper Equations 1–6):

| Loss | Equation | Weight (λ) | Purpose |
|---|---|---|---|
| **Cross-Entropy Segmentation** | Eq.(1) | λ = 5.0 | Semantic parse map accuracy (20 classes) |
| **L1 Pixel Loss** | Eq.(2) | λ = 10.0 | Pixel-wise garment–body alignment |
| **VGG Perceptual Loss** | Eq.(3) | λ = 1.0 | High-level semantic + texture fidelity |
| **GAN Loss** | Eq.(4) | λ = 1.0 | Realism via conditional PatchGAN discriminator |
| **Total Variation Loss** | Eq.(5) | λ = 0.1 | Spatial smoothness / anti-artifact |
| **Generator Total** | Eq.(6) | — | Weighted sum of all above |

**VGG feature layers used:** `relu1_2`, `relu2_2`, `relu3_4`, `relu4_4`  
**LPIPS backbone for evaluation:** AlexNet (consistent with paper)

### Discriminator Loss
Conditional PatchGAN discriminator conditioned on the agnostic person image. Trained with LSGAN-style MSE loss (real → 1, fake → 0), averaged across real/fake.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 512 × 384 |
| Epochs | 100 |
| Batch size | 4 |
| Optimiser (G + D) | Adam |
| Learning rate | 1e-4 |
| Betas | (0.5, 0.999) |
| LR schedule | CosineAnnealingLR (T_max = epochs) |
| Mixed precision | AMP (autocast + GradScaler) |
| Gradient clipping | max norm = 1.0 |
| Seed | 42 |

### Hardware

| Setting | Value |
|---|---|
| GPU (Kaggle run) | Tesla T4 (15.6 GB VRAM) |
| Estimated GPU for full run | NVIDIA RTX 3090 (24 GB) |
| Estimated training time | ~32 hours (100 epochs) |
| Train batches per epoch | 2,911 |
| Total training steps | 291,100 |

### Training Dynamics (Epoch 11th — observed)


| Step          | G Loss    | D Loss    | L1        | VGG       | CE        |
| ------------- | --------- | --------- | --------- | --------- | --------- |
| 200           | 3.412     | 0.102     | 0.102     | 1.163     | 0.110     |
| 400           | 3.292     | 0.113     | 0.098     | 1.144     | 0.107     |
| 800           | 3.283     | 0.111     | 0.096     | 1.142     | 0.106     |
| 1200          | 3.273     | 0.111     | 0.096     | 1.142     | 0.105     |
| 1600          | 3.283     | 0.109     | 0.097     | 1.144     | 0.105     |
| 2000          | 3.288     | 0.108     | 0.096     | 1.141     | 0.106     |
| 2600          | 3.297     | 0.106     | 0.096     | 1.140     | 0.107     |
| **Epoch End** | **3.296** | **0.106** | **0.096** | **1.140** | **0.106** |

---

