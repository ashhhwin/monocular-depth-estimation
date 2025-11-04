
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json
import os


class GeometricDistanceDataset(Dataset):
    """
    Dataset for 3-branch model: full_image + car_patch + geometric_features
    """
    
    def __init__(self, detection_json_path, image_folder, 
                 full_img_size=(224, 224), patch_size=(64, 64),
                 normalize=True):
        """
        Args:
            detection_json_path: Path to detection JSON (e.g., train_detections.json)
            image_folder: Path to folder with images (e.g., Train/)
            full_img_size: Size to resize full images
            patch_size: Size to resize car patches
            normalize: Apply ImageNet normalization
        """
        self.image_folder = image_folder
        self.full_img_size = full_img_size
        self.patch_size = patch_size
        
        # Load detection JSON
        with open(detection_json_path, 'r') as f:
            detection_data = json.load(f)
        
        # Filter: only images with lead vehicle in ROI
        self.samples = []
        for img_name, detection in detection_data.items():
            if detection.get('lead_vehicle') and detection['lead_vehicle'].get('in_roi'):
                img_path = os.path.join(image_folder, img_name)
                
                # Check if image exists
                if os.path.exists(img_path):
                    self.samples.append({
                        'image_path': img_path,
                        'bbox': detection['lead_vehicle']['bbox'],
                        'ground_truth': detection.get('ground_truth_distance'),
                        'img_width': detection['metadata']['image_dimensions']['width'],
                        'img_height': detection['metadata']['image_dimensions']['height']
                    })
        
        print(f"✅ GeometricDataset: Loaded {len(self.samples)} samples")
        
        # Normalization transform
        if normalize:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Full image
        full_img = cv2.resize(image, self.full_img_size)
        full_img = full_img.astype(np.float32) / 255.0  # [0, 1]
        full_img = torch.from_numpy(full_img).permute(2, 0, 1)  # HWC -> CHW
        
        # Car patch
        x1, y1, x2, y2 = map(int, sample['bbox'])
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        patch = image[y1:y2, x1:x2]
        patch = cv2.resize(patch, self.patch_size)
        patch = patch.astype(np.float32) / 255.0  # [0, 1]
        patch = torch.from_numpy(patch).permute(2, 0, 1)  # HWC -> CHW
        
        # Apply normalization
        if self.normalize:
            full_img = self.normalize(full_img)
            patch = self.normalize(patch)
        
        # === NEW: Calculate 10 geometric features (original bbox coordinates) ===
        img_width = sample['img_width']
        img_height = sample['img_height']
        
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = sample['bbox']

        bbox_width = bbox_x2 - bbox_x1
        bbox_height = bbox_y2 - bbox_y1
        bbox_center_x = (bbox_x1 + bbox_x2) / 2
        bbox_center_y = (bbox_y1 + bbox_y2) / 2
        bbox_aspect_ratio = bbox_height / (bbox_width + 1e-6)

        feature_1 = bbox_width / img_width
        feature_2 = bbox_height / img_height
        feature_3 = bbox_y1 / img_height
        feature_4 = bbox_y2 / img_height
        feature_5 = bbox_center_x / img_width
        feature_6 = bbox_center_y / img_height
        feature_7 = bbox_aspect_ratio
        # Features 8-10: Calibrated features using camera matrix
        fx = 2429.865965
        fy = 2424.492001
        cx = 1192.584876
        cy = 1015.978074
        AVG_CAR_HEIGHT = 1.5
        
        # Feature 8: Distance estimate from bbox height
        distance_estimate = (AVG_CAR_HEIGHT * fy) / (bbox_height + 1e-6)
        distance_estimate_norm = np.clip(distance_estimate / 100.0, 0, 1)
        
        # Feature 9: Vertical viewing angle
        vertical_angle = np.arctan2(bbox_y2 - cy, fy)
        vertical_angle_norm = (vertical_angle + np.pi/2) / np.pi
        
        # Feature 10: Angular height
        angular_height = 2 * np.arctan(bbox_height / (2 * fy))
        angular_height_norm = angular_height / (np.pi/2)
        
        geometric_features = torch.tensor([
            feature_1,
            feature_2,
            feature_3,
            feature_4,
            feature_5,
            feature_6,
            feature_7,
            distance_estimate_norm,
            vertical_angle_norm,
            angular_height_norm
        ], dtype=torch.float32)

        # Ground truth
        ground_truth = torch.tensor(sample['ground_truth'], dtype=torch.float32)
        
        return {
            'full_image': full_img,
            'car_patch': patch,
            'geometric': geometric_features
        }, ground_truth

class DepthDistanceDataset(Dataset):
    """
    Dataset for 2-branch model: car_patch + depth_features
    """
    
    def __init__(self, depth_json_path, image_folder,
                 patch_size=(64, 64), normalize=True):
        """
        Args:
            depth_json_path: Path to depth features JSON (e.g., train_depth_features.json)
            image_folder: Path to folder with images
            patch_size: Size to resize car patches
            normalize: Apply ImageNet normalization
        """
        self.image_folder = image_folder
        self.patch_size = patch_size
        
        # Load depth JSON
        with open(depth_json_path, 'r') as f:
            depth_data = json.load(f)
        
        # Create samples
        self.samples = []
        for img_name, data in depth_data.items():
            img_path = os.path.join(image_folder, img_name)
            
            # Check if image exists
            if os.path.exists(img_path):
                self.samples.append({
                    'image_path': img_path,
                    'bbox': data['bbox'],
                    'depth_features': data['depth_features'],
                    'ground_truth': data['ground_truth_distance']
                })
        
        print(f"✅ DepthDataset: Loaded {len(self.samples)} samples")
        
        # Normalization transform
        if normalize:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Car patch
        x1, y1, x2, y2 = map(int, sample['bbox'])
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        patch = image[y1:y2, x1:x2]
        patch = cv2.resize(patch, self.patch_size)
        patch = patch.astype(np.float32) / 255.0  # [0, 1]
        patch = torch.from_numpy(patch).permute(2, 0, 1)  # HWC -> CHW
        
        # Apply normalization
        if self.normalize:
            patch = self.normalize(patch)
        
        # Depth features
        depth_features = torch.tensor(sample['depth_features'], dtype=torch.float32)
        
        # Ground truth
        ground_truth = torch.tensor(sample['ground_truth'], dtype=torch.float32)
        
        return {
            'car_patch': patch,
            'depth_features': depth_features
        }, ground_truth


def create_geometric_dataloaders(
    detection_json_dir,
    image_base_dir,
    batch_size=32,
    num_workers=4,
    full_img_size=(224, 224),
    patch_size=(64, 64),
    shuffle_train=True,
    normalize=True
):
    """
    Create train, val, test dataloaders for geometric features model
    
    Args:
        detection_json_dir: Directory containing detection JSONs
        image_base_dir: Base directory containing Train/Val/Test folders
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        full_img_size: Size for full images
        patch_size: Size for car patches
        shuffle_train: Shuffle training data
        normalize: Apply ImageNet normalization
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*70)
    print("CREATING GEOMETRIC DATALOADERS")
    print("="*70)
    
    # Create datasets
    train_dataset = GeometricDistanceDataset(
        detection_json_path=os.path.join(detection_json_dir, 'train_detections.json'),
        image_folder=os.path.join(image_base_dir, 'Train'),
        full_img_size=full_img_size,
        patch_size=patch_size,
        normalize=normalize
    )
    
    val_dataset = GeometricDistanceDataset(
        detection_json_path=os.path.join(detection_json_dir, 'val_detections.json'),
        image_folder=os.path.join(image_base_dir, 'Val'),
        full_img_size=full_img_size,
        patch_size=patch_size,
        normalize=normalize
    )
    
    test_dataset = GeometricDistanceDataset(
        detection_json_path=os.path.join(detection_json_dir, 'test_detections.json'),
        image_folder=os.path.join(image_base_dir, 'Test'),
        full_img_size=full_img_size,
        patch_size=patch_size,
        normalize=normalize
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ Dataloaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def create_depth_dataloaders(
    depth_json_dir,
    image_base_dir,
    batch_size=32,
    num_workers=4,
    patch_size=(64, 64),
    shuffle_train=True,
    normalize=True
):
    """
    Create train, val, test dataloaders for depth features model
    
    Args:
        depth_json_dir: Directory containing depth feature JSONs
        image_base_dir: Base directory containing Train/Val/Test folders
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        patch_size: Size for car patches
        shuffle_train: Shuffle training data
        normalize: Apply ImageNet normalization
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*70)
    print("CREATING DEPTH DATALOADERS")
    print("="*70)
    
    # Create datasets
    train_dataset = DepthDistanceDataset(
        depth_json_path=os.path.join(depth_json_dir, 'train_depth_features.json'),
        image_folder=os.path.join(image_base_dir, 'Train'),
        patch_size=patch_size,
        normalize=normalize
    )
    
    val_dataset = DepthDistanceDataset(
        depth_json_path=os.path.join(depth_json_dir, 'val_depth_features.json'),
        image_folder=os.path.join(image_base_dir, 'Val'),
        patch_size=patch_size,
        normalize=normalize
    )
    
    test_dataset = DepthDistanceDataset(
        depth_json_path=os.path.join(depth_json_dir, 'test_depth_features.json'),
        image_folder=os.path.join(image_base_dir, 'Test'),
        patch_size=patch_size,
        normalize=normalize
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ Dataloaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

'''
# Example usage
if __name__ == "__main__":
    # Example 1: Geometric features dataloaders
    train_gen, val_gen, test_gen = create_geometric_dataloaders(
        detection_json_dir='~/gcs-bucket/lead_vehicle_features',
        image_base_dir='~/gcs-bucket/2025_UChicago_Distance_Data/final_preprocessed',
        batch_size=32,
        num_workers=4
    )
    
    # Test batch
    batch = next(iter(train_gen))
    inputs, targets = batch
    print(f"\nGeometric batch shapes:")
    print(f"  Full image: {inputs['full_image'].shape}")
    print(f"  Car patch: {inputs['car_patch'].shape}")
    print(f"  Geometric: {inputs['geometric'].shape}")
    print(f"  Targets: {targets.shape}")
    
    # Example 2: Depth features dataloaders
    train_gen, val_gen, test_gen = create_depth_dataloaders(
        depth_json_dir='~/gcs-bucket/depth_features',
        image_base_dir='~/gcs-bucket/2025_UChicago_Distance_Data/final_preprocessed',
        batch_size=32,
        num_workers=4
    )
    
    # Test batch
    batch = next(iter(train_gen))
    inputs, targets = batch
    print(f"\nDepth batch shapes:")
    print(f"  Car patch: {inputs['car_patch'].shape}")
    print(f"  Depth features: {inputs['depth_features'].shape}")
    print(f"  Targets: {targets.shape}")


'''
