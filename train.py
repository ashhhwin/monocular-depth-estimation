# ===============================================================
# UNIFIED TRAINING PIPELINE: Lead Vehicle Detection + DepthPro + CNN Training
# ===============================================================
# This script:
# 1. Detects lead vehicles across multiple input folders
# 2. Extracts depth features using DepthPro
# 3. Trains an EfficientNet-B0 model with attention fusion
# 4. Uses K-Fold cross-validation
# 5. Logs everything to TensorBoard
# 6. Implements early stopping and best model saving
# ===============================================================

import os
import json
import gc
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
import timm  # For EfficientNet
from sklearn.model_selection import KFold

# Import lead vehicle detector
try:
    from lead_vehicle_detector import LeadVehicleDetector
except ImportError:
    print("ERROR: lead_vehicle_detector.py must be in the same directory!")
    exit(1)


# ===============================================================
# CONFIG
# ===============================================================
class Config:
    # Paths (set via argparse)
    input_folders = []
    output_dir = "training_output"
    
    # Model
    backbone = "efficientnet_b0"
    pretrained = True
    
    # Training
    k_folds = 5
    epochs = 100
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 15
    
    # Data
    val_split = 0.15  # Used if not doing cross-validation
    image_size = 224
    
    # DepthPro
    cache_depth = True
    depth_cache_dir = "depth_cache"
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    
    # Lead vehicle detector
    yolo_model = "yolov8l.pt"
    yolo_conf = 0.25


# ===============================================================
# ATTENTION MODULES
# ===============================================================
class SpatialAttention(nn.Module):
    """Learns to focus on important spatial regions"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class FeatureFusionAttention(nn.Module):
    """Learns to weight importance of visual vs depth features"""
    def __init__(self, visual_dim, scalar_dim):
        super().__init__()
        total_dim = visual_dim + scalar_dim
        self.attention = nn.Sequential(
            nn.Linear(total_dim, total_dim // 4),
            nn.ReLU(),
            nn.Linear(total_dim // 4, total_dim),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features, scalar_features):
        combined = torch.cat([visual_features, scalar_features], dim=1)
        weights = self.attention(combined)
        weighted = combined * weights
        return weighted


# ===============================================================
# MODEL: EfficientNet + Attention + Depth Fusion
# ===============================================================
class DistanceEstimator(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True):
        super().__init__()
        
        # EfficientNet backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        backbone_out_dim = self.backbone.num_features
        
        # Spatial attention on backbone features
        self.spatial_attention = SpatialAttention(backbone_out_dim)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Depth scalar processing
        scalar_dim = 4  # min, mean, median, bottom
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Feature fusion with attention
        self.fusion_attention = FeatureFusionAttention(backbone_out_dim, 64)
        
        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(backbone_out_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, images, depth_scalars):
        # Extract visual features
        features = self.backbone.forward_features(images)
        
        # Apply spatial attention
        features = self.spatial_attention(features)
        
        # Global pooling
        visual_features = self.global_pool(features).flatten(1)
        
        # Process depth scalars
        scalar_features = self.scalar_encoder(depth_scalars)
        
        # Fuse with attention
        fused = self.fusion_attention(visual_features, scalar_features)
        
        # Predict distance
        distance = self.regressor(fused).squeeze(1)
        
        return distance


# ===============================================================
# DATASET
# ===============================================================
class DistanceDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: List of dicts with keys:
            - image_path: str
            - depth_scalars: [min, mean, median, bottom]
            - ground_truth: float
        """
        self.data = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Depth scalars
        scalars = torch.tensor(item['depth_scalars'], dtype=torch.float32)
        
        # Ground truth
        label = torch.tensor(item['ground_truth'], dtype=torch.float32)
        
        return image, scalars, label


# ===============================================================
# STAGE 1: LEAD VEHICLE DETECTION
# ===============================================================
def run_lead_vehicle_detection(input_folders, output_json, config):
    """Detect lead vehicles in all folders and save to single JSON"""
    print("\n" + "="*70)
    print("STAGE 1: LEAD VEHICLE DETECTION")
    print("="*70)
    
    detector = LeadVehicleDetector(
        model_path=config.yolo_model,
        conf_threshold=config.yolo_conf,
        use_adaptive_roi=True,
        device=str(config.device)
    )
    
    all_results = {}
    
    for folder_idx, folder in enumerate(input_folders):
        folder_name = Path(folder).name
        print(f"\nProcessing folder {folder_idx+1}/{len(input_folders)}: {folder}")
        
        # Get all images
        image_paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        print(f"Found {len(image_paths)} images")
        
        for img_path in tqdm(image_paths, desc=f"Detecting vehicles"):
            result, _ = detector.find_lead_vehicle(img_path, visualize=False)
            
            if result is None or result['lead_vehicle'] is None:
                continue
            
            # Create unique key: foldername/filename
            filename = os.path.basename(img_path)
            unique_key = f"{folder_name}/{filename}"
            
            # Extract ground truth from filename (format: seqXXX_distYY.YY_...)
            ground_truth = None
            try:
                if '_dist' in filename:
                    dist_str = filename.split('_dist')[1].split('_')[0]
                    ground_truth = float(dist_str)
            except:
                print(f"Warning: Could not parse ground truth from {filename}")
                continue
            
            if ground_truth is None:
                continue
            
            all_results[unique_key] = {
                'image_path': img_path,
                'ground_truth': ground_truth,
                'lead_bbox': result['lead_vehicle']['bbox'],
                'confidence': result['lead_vehicle']['confidence']
            }
            
            # Clear memory
            del result
            gc.collect()
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Detected lead vehicles in {len(all_results)} images")
    print(f"✓ Results saved to: {output_json}")
    
    return all_results


# ===============================================================
# STAGE 2: DEPTH EXTRACTION WITH CACHING
# ===============================================================
def run_depth_extraction(results_dict, config):
    """Extract depth features using DepthPro with disk caching"""
    print("\n" + "="*70)
    print("STAGE 2: DEPTH FEATURE EXTRACTION")
    print("="*70)
    
    # Create cache directory
    if config.cache_depth:
        os.makedirs(config.depth_cache_dir, exist_ok=True)
    
    # Load DepthPro
    print("Loading DepthPro model...")
    processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    dp_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf")
    dp_model = dp_model.to(config.device).half().eval()
    print("✓ DepthPro ready\n")
    
    processed_data = []
    
    for unique_key, info in tqdm(results_dict.items(), desc="Extracting depth features"):
        img_path = info['image_path']
        bbox = info['lead_bbox']
        ground_truth = info['ground_truth']
        
        # Check cache
        cache_key = unique_key.replace('/', '_').replace('.jpg', '.npy').replace('.png', '.npy')
        cache_path = os.path.join(config.depth_cache_dir, cache_key) if config.cache_depth else None
        
        depth_scalars = None
        
        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            try:
                depth_scalars = np.load(cache_path)
            except:
                pass
        
        # Compute if not cached
        if depth_scalars is None:
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                
                # Run DepthPro
                inputs = processor(images=img, return_tensors="pt").to(config.device)
                for k in inputs:
                    inputs[k] = inputs[k].half()
                
                with torch.no_grad():
                    outputs = dp_model(**inputs)
                
                depth_map = processor.post_process_depth_estimation(
                    outputs, target_sizes=[(img.height, img.width)]
                )[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
                
                # Extract scalars from bbox
                x_min, y_min, x_max, y_max = map(int, bbox)
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(depth_map.shape[1], x_max)
                y_max = min(depth_map.shape[0], y_max)
                
                bbox_depth = depth_map[y_min:y_max, x_min:x_max]
                
                if bbox_depth.size == 0:
                    continue
                
                min_d = float(np.min(bbox_depth))
                mean_d = float(np.mean(bbox_depth))
                median_d = float(np.median(bbox_depth))
                
                # Bottom center depth
                bottom_y = min(y_max, depth_map.shape[0] - 1)
                bottom_x = min((x_min + x_max) // 2, depth_map.shape[1] - 1)
                bottom_d = float(depth_map[bottom_y, bottom_x])
                
                depth_scalars = np.array([min_d, mean_d, median_d, bottom_d], dtype=np.float32)
                
                # Cache it
                if cache_path:
                    np.save(cache_path, depth_scalars)
                
                # Cleanup
                del img, inputs, outputs, depth_map
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing {unique_key}: {e}")
                continue
        
        # Add to dataset
        processed_data.append({
            'image_path': img_path,
            'depth_scalars': depth_scalars.tolist(),
            'ground_truth': ground_truth,
            'unique_key': unique_key
        })
    
    # Cleanup DepthPro
    del dp_model, processor
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n✓ Extracted depth features for {len(processed_data)} images")
    
    return processed_data


# ===============================================================
# STAGE 3: TRAINING WITH K-FOLD CV
# ===============================================================
class Trainer:
    def __init__(self, config):
        self.config = config
        
    def train_single_fold(self, fold, train_loader, val_loader, writer):
        """Train a single fold"""
        
        # Initialize model
        model = DistanceEstimator(
            backbone=self.config.backbone,
            pretrained=self.config.pretrained
        ).to(self.config.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        
        print(f"\n{'='*70}")
        print(f"Training Fold {fold+1}")
        print(f"{'='*70}")
        
        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for images, scalars, labels in train_loader:
                images = images.to(self.config.device)
                scalars = scalars.to(self.config.device)
                labels = labels.to(self.config.device)
                
                optimizer.zero_grad()
                predictions = model(images, scalars)
                loss = criterion(predictions, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                train_mae += torch.abs(predictions - labels).sum().item()
            
            train_loss /= len(train_loader.dataset)
            train_mae /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for images, scalars, labels in val_loader:
                    images = images.to(self.config.device)
                    scalars = scalars.to(self.config.device)
                    labels = labels.to(self.config.device)
                    
                    predictions = model(images, scalars)
                    loss = criterion(predictions, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    val_mae += torch.abs(predictions - labels).sum().item()
            
            val_loss /= len(val_loader.dataset)
            val_mae /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Step scheduler
            scheduler.step()
            
            # TensorBoard logging
            global_step = fold * self.config.epochs + epoch
            writer.add_scalar(f'Fold{fold+1}/Train/Loss', train_loss, global_step)
            writer.add_scalar(f'Fold{fold+1}/Train/MAE', train_mae, global_step)
            writer.add_scalar(f'Fold{fold+1}/Val/Loss', val_loss, global_step)
            writer.add_scalar(f'Fold{fold+1}/Val/MAE', val_mae, global_step)
            writer.add_scalar(f'Fold{fold+1}/LearningRate', optimizer.param_groups[0]['lr'], global_step)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | "
                      f"Val Loss: {val_loss:.4f} MAE: {val_mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save best model for this fold
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_loss': best_val_loss,
                    'val_mae': val_mae
                }, os.path.join(self.config.output_dir, f'best_model_fold{fold+1}.pth'))
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        print(f"\nFold {fold+1} completed | Best Val Loss: {best_val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model, best_val_loss, train_losses, val_losses
    
    def run_cross_validation(self, dataset_list, writer):
        """Run K-fold cross validation"""
        
        print("\n" + "="*70)
        print("STAGE 3: MODEL TRAINING WITH K-FOLD CROSS-VALIDATION")
        print("="*70)
        print(f"Total samples: {len(dataset_list)}")
        print(f"K-folds: {self.config.k_folds}")
        print(f"Backbone: {self.config.backbone}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Device: {self.config.device}\n")
        
        # Shuffle data
        random.shuffle(dataset_list)
        
        # K-Fold split
        kfold = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_list)):
            # Create datasets
            train_data = [dataset_list[i] for i in train_ids]
            val_data = [dataset_list[i] for i in val_ids]
            
            train_dataset = DistanceDataset(train_data, transform=train_transform)
            val_dataset = DistanceDataset(val_data, transform=val_transform)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            # Train fold
            model, best_val_loss, train_losses, val_losses = self.train_single_fold(
                fold, train_loader, val_loader, writer
            )
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            })
            
            # Cleanup
            del model, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
        
        # Summary statistics
        avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
        std_val_loss = np.std([r['best_val_loss'] for r in fold_results])
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION RESULTS")
        print("="*70)
        for result in fold_results:
            print(f"Fold {result['fold']}: Best Val Loss = {result['best_val_loss']:.4f}")
        print(f"\nAverage Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        print("="*70)
        
        # Log to TensorBoard
        writer.add_text('CrossValidation/Summary', 
                       f"Avg Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        
        # Save summary
        summary = {
            'k_folds': self.config.k_folds,
            'avg_val_loss': float(avg_val_loss),
            'std_val_loss': float(std_val_loss),
            'fold_results': fold_results
        }
        
        with open(os.path.join(self.config.output_dir, 'cv_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return fold_results


# ===============================================================
# MAIN PIPELINE
# ===============================================================
def main():
    parser = argparse.ArgumentParser(description='Train distance estimation model')
    parser.add_argument('--input_folders', nargs='+', required=True,
                       help='One or more input folders containing images')
    parser.add_argument('--output_dir', default='training_output',
                       help='Output directory for all results')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs per fold')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--skip_detection', action='store_true',
                       help='Skip detection if results.json already exists')
    parser.add_argument('--skip_depth', action='store_true',
                       help='Skip depth extraction if cache exists')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.input_folders = args.input_folders
    config.output_dir = args.output_dir
    config.k_folds = args.k_folds
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Paths
    results_json = os.path.join(config.output_dir, 'detection_results.json')
    depth_data_json = os.path.join(config.output_dir, 'depth_data.json')
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'tensorboard'))
    
    # Stage 1: Lead Vehicle Detection
    if args.skip_detection and os.path.exists(results_json):
        print(f"Skipping detection, loading from {results_json}")
        with open(results_json, 'r') as f:
            results_dict = json.load(f)
    else:
        results_dict = run_lead_vehicle_detection(
            config.input_folders,
            results_json,
            config
        )
    
    # Stage 2: Depth Extraction
    if args.skip_depth and os.path.exists(depth_data_json):
        print(f"Skipping depth extraction, loading from {depth_data_json}")
        with open(depth_data_json, 'r') as f:
            dataset_list = json.load(f)
    else:
        dataset_list = run_depth_extraction(results_dict, config)
        
        # Save depth data
        with open(depth_data_json, 'w') as f:
            json.dump(dataset_list, f, indent=2)
        print(f"✓ Depth data saved to: {depth_data_json}")
    
    # Stage 3: Training
    trainer = Trainer(config)
    fold_results = trainer.run_cross_validation(dataset_list, writer)
    
    # Close TensorBoard
    writer.close()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Output directory: {config.output_dir}")
    print(f"  - Detection results: {results_json}")
    print(f"  - Depth data: {depth_data_json}")
    print(f"  - Model checkpoints: best_model_fold*.pth")
    print(f"  - TensorBoard logs: tensorboard/")
    print(f"  - CV summary: cv_summary.json")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir {os.path.join(config.output_dir, 'tensorboard')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()