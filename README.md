# Hierarchical encoder-decoder framework with dynamic loss for fine-grained lithology identification from well-log data

A deep learning framework for hierarchical lithology identification from well-log data, implementing attention-enhanced feature encoding and hierarchical fusion decoding.

## Overview

This repository contains the implementation of a hierarchical encoder-decoder framework designed to identify lithology from geophysical well-log data across three taxonomic levels (primary, secondary, and tertiary). The model addresses key challenges in automated lithology identification:

- **High-dimensional feature extraction** from complex geophysical logs
- **Hierarchical taxonomy modeling** respecting geological classification structure
- **Severe class imbalance** in lithology datasets

## Key Features

- **Attention-Enhanced Feature Encoder (AEFE)**: Multi-stream parallel processing with spatial and channel attention mechanisms
- **Hierarchical Fusion Decoder**: Task-specific branches with learnable gates ensuring geological consistency
- **Dynamic Hierarchical Loss**: Combines focal loss, consistency regularization, and AUC optimization
- **Class Imbalance Handling**: Hierarchical weighted sampling and adaptive loss weighting

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 1.10.1
- CUDA (optional, for GPU acceleration)

## Quick Start

### Training

```bash
python train.py \
    --data_path /path/to/data \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num_primary 3 \
    --num_secondary 6 \
    --num_tertiary 12
```

## Project Structure

```
lithology-identification/
├── models/              # Model architecture
│   ├── attention.py     # Spatial and channel attention
│   ├── blocks.py        # Convolutional blocks
│   ├── hierarchical_model.py  # AEFE + hierarchical fusion decoder + heads
│   └── baselines.py     # baselines
├── losses/              # Loss functions
│   ├── focal_loss.py    # Focal loss 
│   └── hierarchical_loss.py  # Dynamic hierarchical loss 
├── data/                # Data utilities
│   ├── preprocessing.py # Hampel filter, resampling, Min-Max scaler, CSV parsing
│   └── samplers.py      # Hierarchical weighted sampler
├── utils/               # Training utilities
│   ├── training_utils.py     # Four-stage curriculum training, checkpointing
│   └── metrics.py       # Macro metrics, absent-class leakage, confidence intervals
├── experiments/        
├── train.py             # Training script
├── requirements.txt
└── README.md
```

## Acknowledgments

- Research Institution of Petroleum Exploration & Development, PetroChina
- School of Geosciences, Yangtze University

