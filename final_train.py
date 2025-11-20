# ===============================================================
# TRAINING PIPELINE: ConvNeXt-Large + 4 Depth + 1 Geometric Features
# ===============================================================

import os
import json
import gc
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
import timm

try:
    from lead_vehicle_detector import LeadVehicleDetector
except ImportError:
    print("ERROR: lead_vehicle_detector.py must be in the same directory!")
    exit(1)

try:
    from preprocessing import preprocess_image
except ImportError:
    print("ERROR: preprocessing.py must be in the same directory!")
    exit(1)


# ===============================================================
# CONFIG
# ===============================================================
class Config:
    # Fixed train/val/test split
    train_folders = [
        'DistDS-5_Nashville1',
        'DistDS-5_Nashville2', 
        'DistDS-6_Nashville3',
        'DistDS-7_Nashville4',
        'LA1',
        'LA2',
        'I_55_3',
        'I_55_4'
    ]
    
    val_folders = [
        'I_55_1',
        'I_55_5'
    ]
    
    test_folders = [
        'DistDS-2_UCRS_w2e',
        'DistDS-3_UCRS_e2w',
        'DistDS-4_uic',
        'DistDS01_Chattanooga_April06',
        'I_55_2'
    ]
    
    # Paths
    data_root = ""  # Will be set via argparse
    output_dir = "training_output"
    
    # Model
    backbone = "convnext_large"
    pretrained = True
    
    # Training
    epochs = 100
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 15
    
    # Data
    image_size = 384  # ConvNeXt-Large native size
    crop_padding = 0.15
    
    # Camera calibration
    focal_length_y = 2424.492001
    avg_car_height = 1.5
    
    # Cache
    cache_preprocessed = True
    preprocessed_cache_dir = "preprocessed_cache"
    cache_depth = True
    depth_cache_dir = "depth_cache"
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    
    # YOLO
    yolo_model = "yolov8l.pt"
    yolo_conf = 0.25


# ===============================================================
# ATTENTION MODULES
# ===============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = (avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([max_pool, avg_pool], dim=1)))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, depth_dim):
        super().__init__()
        self.query = nn.Linear(visual_dim, 128)
        self.key = nn.Linear(depth_dim, 128)
        self.value = nn.Linear(depth_dim, 128)
        self.scale = np.sqrt(128)
        
    def forward(self, visual_features, depth_features):
        Q = self.query(visual_features)
        K = self.key(depth_features)
        V = self.value(depth_features)
        
        attention = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        attended = torch.matmul(attention, V)
        
        return torch.cat([visual_features, attended], dim=1)


# ===============================================================
# DEPTH ENCODER
# ===============================================================
class DepthEncoder(nn.Module):
    def __init__(self, num_features=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)


# ===============================================================
# MODEL: ConvNeXt-Large
# ===============================================================
class DistanceEstimator(nn.Module):
    def __init__(self, backbone="convnext_large", pretrained=True):
        super().__init__()
        
        # ConvNeXt-Large backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension (1536 for ConvNeXt-Large)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 384, 384)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        # Attention on visual features
        self.attention = CBAM(self.feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Visual feature projection
        self.visual_projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Depth encoder (5 features: 4 depth + 1 geometric)
        self.depth_encoder = DepthEncoder(num_features=5)
        
        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(512, 256)
        
        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(512 + 128, 256),
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
        
        # Apply attention
        features = self.attention(features)
        
        # Global pooling
        pooled = self.global_pool(features).flatten(1)
        
        # Project visual features
        visual_features = self.visual_projection(pooled)
        
        # Process depth features
        depth_features = self.depth_encoder(depth_scalars)
        
        # Cross-attention fusion
        fused = self.cross_attention(visual_features, depth_features)
        
        # Predict distance
        distance = self.regressor(fused).squeeze(1)
        
        return distance


# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================
def crop_lead_vehicle(image, bbox, padding=0.15):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    x1 = max(0, int(x1 - w * padding))
    y1 = max(0, int(y1 - h * padding))
    x2 = min(image.width, int(x2 + w * padding))
    y2 = min(image.height, int(y2 + h * padding))
    
    return image.crop((x1, y1, x2, y2))


def extract_depth_features(depth_map, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(depth_map.shape[1], x_max)
    y_max = min(depth_map.shape[0], y_max)
    
    bbox_depth = depth_map[y_min:y_max, x_min:x_max]
    
    if bbox_depth.size == 0:
        return None
    
    min_d = float(np.min(bbox_depth))
    median_d = float(np.median(bbox_depth))
    p10 = float(np.percentile(bbox_depth, 10))
    
    bottom_y = min(y_max - 1, depth_map.shape[0] - 1)
    center_x = (x_min + x_max) // 2
    center_x = min(center_x, depth_map.shape[1] - 1)
    bottom_center_d = float(depth_map[bottom_y, center_x])
    
    return np.array([min_d, median_d, p10, bottom_center_d], dtype=np.float32)


def compute_geometric_distance(bbox, focal_length_y, avg_car_height):
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    
    if bbox_height <= 0:
        return 0.0
    
    distance_estimate = (avg_car_height * focal_length_y) / bbox_height
    return float(distance_estimate)


def parse_distance_from_filename(filename):
    try:
        if '_dist' in filename:
            dist_str = filename.split('_dist')[1].split('_')[0]
            return float(dist_str)
    except:
        pass
    return None


# ===============================================================
# DATASET
# ===============================================================
class DistanceDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_path = item['cropped_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 5 scalar features: 4 depth + 1 geometric
        scalars = torch.tensor(
            item['depth_scalars'] + [item['geometric_distance']], 
            dtype=torch.float32
        )
        
        label = torch.tensor(item['ground_truth'], dtype=torch.float32)
        
        return image, scalars, label


# ===============================================================
# LEAD VEHICLE DETECTION
# ===============================================================
def run_lead_vehicle_detection(input_folders, output_json, config):
    print("\n" + "="*70)
    print("LEAD VEHICLE DETECTION")
    print("="*70)
    
    detector = LeadVehicleDetector(
        model_path=config.yolo_model,
        conf_threshold=config.yolo_conf,
        use_adaptive_roi=True,
        device=str(config.device)
    )
    
    all_results = {}
    
    for folder in input_folders:
        folder_path = os.path.join(config.data_root, folder)
        folder_name = Path(folder_path).name
        print(f"\nProcessing: {folder}")
        
        image_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        print(f"Found {len(image_paths)} images")
        
        for img_path in tqdm(image_paths, desc="Detecting"):
            result, _ = detector.find_lead_vehicle(img_path, visualize=False)
            
            if result is None or result['lead_vehicle'] is None:
                continue
            
            filename = os.path.basename(img_path)
            unique_key = f"{folder_name}/{filename}"
            
            ground_truth = parse_distance_from_filename(filename)
            
            if ground_truth is None:
                continue
            
            all_results[unique_key] = {
                'image_path': img_path,
                'ground_truth': ground_truth,
                'lead_bbox': result['lead_vehicle']['bbox'],
                'confidence': result['lead_vehicle']['confidence']
            }
            
            del result
            gc.collect()
    
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetected lead vehicles in {len(all_results)} images")
    
    return all_results


# ===============================================================
# PREPROCESSING
# ===============================================================
def run_image_preprocessing(results_dict, config):
    print("\n" + "="*70)
    print("IMAGE PREPROCESSING + CROPPING")
    print("="*70)
    
    if config.cache_preprocessed:
        os.makedirs(config.preprocessed_cache_dir, exist_ok=True)
        cropped_dir = os.path.join(config.preprocessed_cache_dir, "cropped")
        os.makedirs(cropped_dir, exist_ok=True)
    
    preprocessed_results = {}
    
    for unique_key, info in tqdm(results_dict.items(), desc="Preprocessing"):
        img_path = info['image_path']
        bbox = info['lead_bbox']
        
        cache_filename = unique_key.replace('/', '_')
        preprocessed_path = os.path.join(config.preprocessed_cache_dir, cache_filename)
        cropped_path = os.path.join(cropped_dir, cache_filename)
        
        if (config.cache_preprocessed and 
            os.path.exists(preprocessed_path) and 
            os.path.exists(cropped_path)):
            pass
        else:
            try:
                original_img = cv2.imread(img_path)
                if original_img is None:
                    continue
                
                preprocessed_img = preprocess_image(original_img)
                
                if config.cache_preprocessed:
                    cv2.imwrite(preprocessed_path, preprocessed_img)
                
                preprocessed_pil = Image.fromarray(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
                cropped_img = crop_lead_vehicle(preprocessed_pil, bbox, padding=config.crop_padding)
                
                if config.cache_preprocessed:
                    cropped_img.save(cropped_path)
                
                del original_img, preprocessed_img, preprocessed_pil, cropped_img
                gc.collect()
                
            except Exception as e:
                print(f"\nError preprocessing {unique_key}: {e}")
                continue
        
        preprocessed_results[unique_key] = {
            **info,
            'preprocessed_path': preprocessed_path,
            'cropped_path': cropped_path
        }
    
    print(f"\nPreprocessed {len(preprocessed_results)} images")
    
    return preprocessed_results


# ===============================================================
# DEPTH EXTRACTION
# ===============================================================
def run_depth_extraction(results_dict, config):
    print("\n" + "="*70)
    print("DEPTH FEATURE EXTRACTION")
    print("="*70)
    
    if config.cache_depth:
        os.makedirs(config.depth_cache_dir, exist_ok=True)
    
    print("Loading DepthPro model...")
    processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    dp_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf")
    dp_model = dp_model.to(config.device).half().eval()
    
    processed_data = []
    
    for unique_key, info in tqdm(results_dict.items(), desc="Extracting depth"):
        preprocessed_path = info['preprocessed_path']
        bbox = info['lead_bbox']
        ground_truth = info['ground_truth']
        
        cache_key = unique_key.replace('/', '_').replace('.jpg', '.npy').replace('.png', '.npy')
        cache_path = os.path.join(config.depth_cache_dir, cache_key) if config.cache_depth else None
        
        depth_scalars = None
        
        if cache_path and os.path.exists(cache_path):
            try:
                depth_scalars = np.load(cache_path)
                if depth_scalars.shape[0] != 4:
                    depth_scalars = None
            except:
                pass
        
        if depth_scalars is None:
            try:
                img = Image.open(preprocessed_path).convert('RGB')
                
                inputs = processor(images=img, return_tensors="pt").to(config.device)
                for k in inputs:
                    inputs[k] = inputs[k].half()
                
                with torch.no_grad():
                    outputs = dp_model(**inputs)
                
                depth_map = processor.post_process_depth_estimation(
                    outputs, target_sizes=[(img.height, img.width)]
                )[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
                
                depth_scalars = extract_depth_features(depth_map, bbox)
                
                if depth_scalars is None:
                    continue
                
                if cache_path:
                    np.save(cache_path, depth_scalars)
                
                del img, inputs, outputs, depth_map
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing {unique_key}: {e}")
                continue
        
        geometric_distance = compute_geometric_distance(
            bbox, config.focal_length_y, config.avg_car_height
        )
        
        processed_data.append({
            'image_path': info['image_path'],
            'preprocessed_path': preprocessed_path,
            'cropped_path': info['cropped_path'],
            'depth_scalars': depth_scalars.tolist(),
            'geometric_distance': geometric_distance,
            'ground_truth': ground_truth,
            'unique_key': unique_key
        })
    
    del dp_model, processor
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\nExtracted features for {len(processed_data)} images")
    
    return processed_data


# ===============================================================
# EVALUATION
# ===============================================================
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, scalars, labels in data_loader:
            images = images.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device)
            
            predictions = model(images, scalars)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item() * images.size(0)
            total_mae += torch.abs(predictions - labels).sum().item()
            total_samples += images.size(0)
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    # Calculate RMSE
    predictions_np = np.array(all_predictions)
    labels_np = np.array(all_labels)
    rmse = np.sqrt(np.mean((predictions_np - labels_np) ** 2))
    
    return avg_loss, avg_mae, rmse


# ===============================================================
# TRAINING
# ===============================================================
def train_model(model, train_loader, val_loader, test_loader, config, writer):
    criterion = nn.HuberLoss(delta=1.0)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for images, scalars, labels in train_loader:
            images = images.to(config.device)
            scalars = scalars.to(config.device)
            labels = labels.to(config.device)
            
            optimizer.zero_grad()
            predictions = model(images, scalars)
            loss = criterion(predictions, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_mae += torch.abs(predictions - labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        
        # Validation
        val_loss, val_mae, val_rmse = evaluate_model(model, val_loader, criterion, config.device)
        
        scheduler.step()
        
        # TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/MAE', train_mae, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/MAE', val_mae, epoch)
        writer.add_scalar('Val/RMSE', val_rmse, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | "
                  f"Val Loss: {val_loss:.4f} MAE: {val_mae:.4f} RMSE: {val_rmse:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'val_loss': best_val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'config': {
                    'backbone': config.backbone,
                    'image_size': config.image_size,
                    'num_depth_features': 4,
                    'has_geometric_feature': True
                }
            }, os.path.join(config.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation on all sets
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    train_loss, train_mae, train_rmse = evaluate_model(model, train_loader, criterion, config.device)
    val_loss, val_mae, val_rmse = evaluate_model(model, val_loader, criterion, config.device)
    test_loss, test_mae, test_rmse = evaluate_model(model, test_loader, criterion, config.device)
    
    print(f"\nTrain - Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f}")
    print(f"Val   - Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")
    print(f"Test  - Loss: {test_loss:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")
    
    results = {
        'train': {'loss': float(train_loss), 'mae': float(train_mae), 'rmse': float(train_rmse)},
        'val': {'loss': float(val_loss), 'mae': float(val_mae), 'rmse': float(val_rmse)},
        'test': {'loss': float(test_loss), 'mae': float(test_mae), 'rmse': float(test_rmse)}
    }
    
    with open(os.path.join(config.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return model


# ===============================================================
# MAIN
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='Root directory containing all folders')
    parser.add_argument('--output_dir', default='training_output')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop_padding', type=float, default=0.15)
    parser.add_argument('--skip_detection', action='store_true')
    parser.add_argument('--skip_preprocessing', action='store_true')
    parser.add_argument('--skip_depth', action='store_true')
    
    args = parser.parse_args()
    
    config = Config()
    config.data_root = args.data_root
    config.output_dir = args.output_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.crop_padding = args.crop_padding
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # JSON paths
    train_json = os.path.join(config.output_dir, 'train_detection.json')
    val_json = os.path.join(config.output_dir, 'val_detection.json')
    test_json = os.path.join(config.output_dir, 'test_detection.json')
    
    train_preprocessed_json = os.path.join(config.output_dir, 'train_preprocessed.json')
    val_preprocessed_json = os.path.join(config.output_dir, 'val_preprocessed.json')
    test_preprocessed_json = os.path.join(config.output_dir, 'test_preprocessed.json')
    
    train_data_json = os.path.join(config.output_dir, 'train_data.json')
    val_data_json = os.path.join(config.output_dir, 'val_data.json')
    test_data_json = os.path.join(config.output_dir, 'test_data.json')
    
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'tensorboard'))
    
    print("\n" + "="*70)
    print("DISTANCE ESTIMATION TRAINING")
    print("="*70)
    print(f"Backbone: {config.backbone}")
    print(f"Image size: {config.image_size}x{config.image_size}")
    print(f"Device: {config.device}")
    print(f"\nTrain folders: {len(config.train_folders)}")
    print(f"Val folders: {len(config.val_folders)}")
    print(f"Test folders: {len(config.test_folders)}")
    
    # Detection
    if not args.skip_detection:
        print("\n" + "="*70)
        print("TRAIN SET")
        train_results = run_lead_vehicle_detection(config.train_folders, train_json, config)
        
        print("\n" + "="*70)
        print("VAL SET")
        val_results = run_lead_vehicle_detection(config.val_folders, val_json, config)
        
        print("\n" + "="*70)
        print("TEST SET")
        test_results = run_lead_vehicle_detection(config.test_folders, test_json, config)
    else:
        with open(train_json, 'r') as f:
            train_results = json.load(f)
        with open(val_json, 'r') as f:
            val_results = json.load(f)
        with open(test_json, 'r') as f:
            test_results = json.load(f)
    
    # Preprocessing
    if not args.skip_preprocessing:
        print("\n" + "="*70)
        print("PREPROCESSING TRAIN SET")
        train_preprocessed = run_image_preprocessing(train_results, config)
        with open(train_preprocessed_json, 'w') as f:
            json.dump(train_preprocessed, f, indent=2)
        
        print("\n" + "="*70)
        print("PREPROCESSING VAL SET")
        val_preprocessed = run_image_preprocessing(val_results, config)
        with open(val_preprocessed_json, 'w') as f:
            json.dump(val_preprocessed, f, indent=2)
        
        print("\n" + "="*70)
        print("PREPROCESSING TEST SET")
        test_preprocessed = run_image_preprocessing(test_results, config)
        with open(test_preprocessed_json, 'w') as f:
            json.dump(test_preprocessed, f, indent=2)
    else:
        with open(train_preprocessed_json, 'r') as f:
            train_preprocessed = json.load(f)
        with open(val_preprocessed_json, 'r') as f:
            val_preprocessed = json.load(f)
        with open(test_preprocessed_json, 'r') as f:
            test_preprocessed = json.load(f)
    
    # Depth extraction
    if not args.skip_depth:
        print("\n" + "="*70)
        print("DEPTH EXTRACTION TRAIN SET")
        train_data = run_depth_extraction(train_preprocessed, config)
        with open(train_data_json, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        print("\n" + "="*70)
        print("DEPTH EXTRACTION VAL SET")
        val_data = run_depth_extraction(val_preprocessed, config)
        with open(val_data_json, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print("\n" + "="*70)
        print("DEPTH EXTRACTION TEST SET")
        test_data = run_depth_extraction(test_preprocessed, config)
        with open(test_data_json, 'w') as f:
            json.dump(test_data, f, indent=2)
    else:
        with open(train_data_json, 'r') as f:
            train_data = json.load(f)
        with open(val_data_json, 'r') as f:
            val_data = json.load(f)
        with open(test_data_json, 'r') as f:
            test_data = json.load(f)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DistanceDataset(train_data, transform=train_transform)
    val_dataset = DistanceDataset(val_data, transform=val_transform)
    test_dataset = DistanceDataset(test_data, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Initialize model
    model = DistanceEstimator(
        backbone=config.backbone,
        pretrained=config.pretrained
    ).to(config.device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Train
    model = train_model(model, train_loader, val_loader, test_loader, config, writer)
    
    writer.close()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Output directory: {config.output_dir}")
    print(f"  - Best model: best_model.pth")
    print(f"  - Results: results.json")
    print(f"  - TensorBoard: tensorboard/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
