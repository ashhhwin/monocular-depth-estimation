# Dashcam Distance Estimation

A comprehensive deep learning pipeline for estimating distances to lead vehicles in dashcam imagery. The system integrates vehicle detection, monocular depth estimation, and a custom regression model with attention mechanisms.

## Overview

This project provides an end-to-end solution for distance estimation from single dashcam images:

1. **Lead Vehicle Detection**: YOLOv8-based detection with adaptive ROI and lane-aware tracking
2. **Depth Feature Extraction**: Apple DepthPro for monocular depth estimation with bounding box-specific metrics
3. **Distance Regression**: EfficientNet-B0 backbone with spatial and feature fusion attention mechanisms

The pipeline supports multiple input directories, automatic ground truth extraction from filenames, k-fold cross-validation, and comprehensive TensorBoard logging.

## Architecture

### Lead Vehicle Detector
- YOLOv8 for vehicle detection
- Adaptive region-of-interest (ROI) based on lane line detection
- Temporal tracking with fallback strategies
- Handles curves and varying road conditions

### Distance Estimation Model
- **Backbone**: EfficientNet-B0 pretrained on ImageNet
- **Spatial Attention**: Learns to focus on relevant image regions
- **Depth Fusion**: Combines visual features with DepthPro scalars (min, mean, median, bottom depth)
- **Feature Fusion Attention**: Weights importance of visual vs depth information
- **Regularization**: Dropout layers, weight decay, gradient clipping

### Training Strategy
- K-fold cross-validation for robust evaluation
- Early stopping to prevent overfitting
- Cosine annealing learning rate schedule
- Mixed precision support for efficient GPU utilization

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
timm>=0.9.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorboard>=2.13.0
Pillow>=10.0.0
tqdm>=4.65.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

Images should be organized in folders with filenames following this convention:

```
seqXXX_distYY.YY_timeZZZZZZZ.jpg
```

Where:
- `seqXXX`: Sequence identifier
- `distYY.YY`: Ground truth distance in meters
- `timeZZZ`: Timestamp (optional)

Example:
```
folder1/seq001_dist18.48_time1743693342.jpg
folder1/seq002_dist19.12_time1743693343.jpg
folder2/seq001_dist15.30_time1743693344.jpg
```

## Usage

### Basic Training

```bash
python train.py --input_folders data/folder1/ data/folder2/ data/folder3/
```

### Advanced Configuration

```bash
python train.py \
  --input_folders data/seq1/ data/seq2/ data/seq3/ \
  --output_dir experiments/run_001 \
  --k_folds 5 \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4
```

### Resume Training (Skip Completed Stages)

```bash
python train.py \
  --input_folders data/seq1/ \
  --skip_detection \
  --skip_depth
```

### Parameters

- `--input_folders`: One or more directories containing dashcam images (required)
- `--output_dir`: Output directory for results (default: `training_output`)
- `--k_folds`: Number of cross-validation folds (default: 5)
- `--epochs`: Maximum epochs per fold (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--skip_detection`: Skip vehicle detection if results already exist
- `--skip_depth`: Skip depth extraction if cache exists

## Output Structure

```
training_output/
├── detection_results.json      # Lead vehicle detections
├── depth_data.json             # Depth features for all images
├── depth_cache/                # Cached depth maps (.npy files)
├── best_model_fold1.pth        # Best model for each fold
├── best_model_fold2.pth
├── ...
├── cv_summary.json             # Cross-validation metrics
└── tensorboard/                # TensorBoard logs
```

## Monitoring Training

View training progress in real-time with TensorBoard:

```bash
tensorboard --logdir training_output/tensorboard
```

Metrics logged per fold:
- Training loss (MSE) and MAE
- Validation loss (MSE) and MAE
- Learning rate schedule
- Cross-validation summary statistics

## Model Performance

The model is evaluated using:
- **Mean Squared Error (MSE)**: Primary loss metric
- **Mean Absolute Error (MAE)**: Average prediction error in meters
- **K-Fold Cross-Validation**: Robust performance estimation across data splits

Overfitting prevention:
- Early stopping with patience
- Dropout regularization (0.3-0.4)
- L2 weight decay
- Gradient clipping
- Cross-validation

## File Descriptions

- `train.py`: Main training pipeline with all three stages
- `lead_vehicle_detector.py`: YOLOv8-based vehicle detection module
- `requirements.txt`: Python dependencies

## Technical Details

### Memory Optimization
- Disk caching for DepthPro depth maps
- Mixed precision (FP16) for depth estimation
- Aggressive memory cleanup between batches
- Gradient checkpointing support

### Robustness Features
- Random data shuffling to prevent temporal leakage
- Unique filename handling across multiple folders
- Graceful handling of missing ground truth
- Error recovery and logging


## License

MIT License

## Acknowledgments

- YOLOv8 by Ultralytics
- Apple DepthPro for monocular depth estimation
- EfficientNet architecture by Google Research
- PyTorch and timm libraries

