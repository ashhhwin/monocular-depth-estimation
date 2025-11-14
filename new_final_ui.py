"""
Dashcam Distance Estimation Pipeline - Dual Model Streamlit UI
Supports both NEW (5-feature FiLM) and OLD (3-branch) models.
- Model 1: FiLMEstimator (EfficientNet-B2 + 5 features + FiLM)
- Model 2: HybridGeometricEstimator (3-branch)

Fixes:
- Integrated new 5-feature FiLM model architecture for Model 1.
- Integrated full 4-step preprocessing pipeline and visualization.
- Fixed use_column_width deprecation warning.
- Removed all emojis.

RUN THIS IN COLAB/JUPYTER:
    !git clone https://github.com/ashhhwin/monocular-depth-estimation.git
    %cd monocular-depth-estimation
    !pip install streamlit torch torchvision opencv-python ultralytics numpy transformers timm
    !streamlit run ui.py --server.port 8501
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import tempfile
import sys
import gc
import time
import logging

# Suppress all warnings
logging.getLogger().setLevel(logging.ERROR)

# Add the cloned repo to Python path
REPO_PATH = os.getcwd()
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Import dependencies
try:
    from lead_vehicle_detector import LeadVehicleDetector
    # We will define preprocessing functions locally
    # from preprocessing import preprocess_image 
    # import config
    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
    import timm
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error(f"Current directory: {os.getcwd()}")
    st.error(f"Files in directory: {os.listdir('.')}")
    st.stop()


# ===============================================================
# PREPROCESSING FUNCTIONS (from preprocessing.py)
# ===============================================================

# --- Preprocessing Constants (from preprocessing.py and config.py) ---
CameraMat = np.array([[2429.865965, 0.0, 1209.084876],
                      [0.0, 2424.492001, 1032.478074],
                      [0.0, 0.0, 1.0]])
DistCoeff = np.zeros((4, 1)) # Assuming no distortion for this example
UNDISTORT_ALPHA = 0.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAMMA = 1.5

# --- Model 1 Geometric Constants (from train.py) ---
FOCAL_LENGTH_Y = 2424.492001
AVG_CAR_HEIGHT = 1.5


def undistort_image(image: np.ndarray) -> np.ndarray:
    """
    Removes lens distortion from an image using camera parameters.
    """
    h, w = image.shape[:2]
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        CameraMat, 
        DistCoeff, 
        (w, h), 
        UNDISTORT_ALPHA, 
        (w, h)
    )
    
    undistorted = cv2.undistort(
        image, 
        CameraMat, 
        DistCoeff, 
        None, 
        new_camera_matrix
    )
    
    if UNDISTORT_ALPHA == 0:
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
    return undistorted


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Applies the "Gray World" white balance algorithm.
    """
    b, g, r = cv2.split(image)
    
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    scale_b = avg_gray / (avg_b + 1e-6)
    scale_g = avg_gray / (avg_g + 1e-6)
    scale_r = avg_gray / (avg_r + 1e-6)
    
    b_balanced = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g_balanced = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r_balanced = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    return cv2.merge([b_balanced, g_balanced, r_balanced])


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    
    l_clahe = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image: np.ndarray) -> np.ndarray:
    """
    Applies non-linear gamma correction to the image.
    """
    inv_gamma = 1.0 / GAMMA
    
    lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    
    return cv2.LUT(image, lut)


def preprocess_image_steps(image_bgr: np.ndarray):
    """
    Runs the full preprocessing pipeline and returns intermediate steps.
    Returns a dictionary of (RGB) steps and the final processed (BGR) image.
    """
    steps = {}
    
    # Step 1: Fix geometric distortion
    processed = undistort_image(image_bgr)
    steps['undistorted'] = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Step 2: Fix color cast
    processed = apply_white_balance(processed)
    steps['white_balanced'] = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Step 3: Enhance local contrast
    processed = apply_clahe(processed)
    steps['clahe_applied'] = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Step 4: Final non-linear brightness adjustment
    processed = apply_gamma_correction(processed)
    steps['gamma_corrected'] = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Return the dictionary of steps and the final BGR image for processing
    return steps, processed


# ============================================================================
# MODEL 1: FiLM ESTIMATOR (NEW 5-FEATURE MODEL)
# ============================================================================

# [!!! START CHANGE !!!]
# Removed CBAM, ChannelAttention, SpatialAttention, CrossAttentionFusion
# Added FiLMLayer and FiLMEstimator

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer.
    Uses scalar_features to generate scale (gamma) and shift (beta)
    parameters to modulate visual_features.
    """
    def __init__(self, scalar_dim, visual_dim):
        super().__init__()
        self.visual_dim = visual_dim
        
        # This MLP generates the gamma and beta parameters
        self.param_generator = nn.Sequential(
            nn.Linear(scalar_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            # Output 2 * visual_dim (one set for gamma, one for beta)
            nn.Linear(256, 2 * visual_dim) 
        )
        
    def forward(self, visual_features, scalar_features):
        # visual_features: [B, visual_dim]
        # scalar_features: [B, scalar_dim]
        
        # Generate params: [B, 2 * visual_dim]
        params = self.param_generator(scalar_features)
        
        # Split into gamma and beta
        gamma, beta = torch.chunk(params, 2, dim=1)
        
        # We add 1.0 to gamma so the default (at init) is an identity
        # transform (gamma=1, beta=0), which is more stable.
        fused_features = (1.0 + gamma) * visual_features + beta
        
        return fused_features

class ReducedDepthEncoder(nn.Module):
    """Encodes 5 features: 4 depth + 1 geometric distance estimate"""
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

class FiLMEstimator(nn.Module):
    """Model 1: Multi-Scale EfficientNet-B2 + 5-Feature FiLM Fusion"""
    def __init__(self, backbone="efficientnet_b2", pretrained=True):
        super().__init__()
        
        # EfficientNet-B2 backbone
        base_model = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        feature_info = base_model.feature_info
        self.backbone = base_model
        
        self.mid_dim = feature_info[-2]['num_chs']
        self.high_dim = feature_info[-1]['num_chs']
        
        # --- CBAM ATTENTION REMOVED ---
        
        # Global pooling for both scales
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fusion of multi-scale features
        self.visual_dim = 512
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.mid_dim + self.high_dim, self.visual_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Reduced depth encoder (5 features: 4 depth + 1 geometric)
        self.scalar_dim = 256
        self.depth_encoder = ReducedDepthEncoder(num_features=5) # Outputs [B, 256]
        
        # --- CROSS-ATTENTION REMOVED ---
        
        # --- NEW FiLM FUSION LAYER ---
        self.film_fusion = FiLMLayer(self.scalar_dim, self.visual_dim)
        
        # Final regression head (inputs the modulated visual_dim)
        self.regressor = nn.Sequential(
            nn.Linear(self.visual_dim, 256), # Input dim is 512
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, images, depth_scalars):
        # Extract multi-scale features
        features = self.backbone(images)
        mid_features = features[-2]
        high_features = features[-1]
        
        # --- NO ATTENTION HERE ---
        
        # Global pooling
        mid_pooled = self.global_pool(mid_features).flatten(1)
        high_pooled = self.global_pool(high_features).flatten(1)
        
        # Fuse multi-scale visual features
        visual_features = self.feature_fusion(torch.cat([mid_pooled, high_pooled], dim=1))
        
        # Process depth features
        depth_features = self.depth_encoder(depth_scalars)
        
        # --- APPLY FiLM FUSION ---
        # The scalars (depth_features) modulate the visuals (visual_features)
        fused = self.film_fusion(visual_features, depth_features)
        
        # Predict distance
        distance = self.regressor(fused).squeeze(1)
        
        return distance

# [!!! END CHANGE !!!]
# ============================================================================


# ============================================================================
# MODEL 2: HYBRID GEOMETRIC ESTIMATOR (OLD MODEL)
# ============================================================================
class HybridGeometricEstimator(nn.Module):
    """Model 2: 3-branch model with EfficientNetB4 + MobileNetV2 + Geometric Features"""
    def __init__(self):
        super(HybridGeometricEstimator, self).__init__()

        efficientnet = models.efficientnet_b4(pretrained=False)
        efficientnet.classifier = nn.Identity()
        self.full_image_backbone = efficientnet
        for param in self.full_image_backbone.parameters():
            param.requires_grad = False
        self.full_image_head = nn.Sequential(
            nn.Linear(1792, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.classifier = nn.Identity()
        self.car_patch_backbone = mobilenet
        for param in self.car_patch_backbone.parameters():
            param.requires_grad = False
        self.car_patch_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.geometric_branch = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, full_image, car_patch, geometric):
        full_features = self.full_image_backbone(full_image)
        full_features = self.full_image_head(full_features)

        patch_features = self.car_patch_backbone(car_patch)
        patch_features = self.car_patch_head(patch_features)

        geo_features = self.geometric_branch(geometric)

        combined = torch.cat([full_features, patch_features, geo_features], dim=1)
        output = self.fusion(combined)

        return output.squeeze(1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def crop_lead_vehicle(image, bbox, padding=0.15):
    """Crop vehicle with padding"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    x1 = max(0, int(x1 - w * padding))
    y1 = max(0, int(y1 - h * padding))
    x2 = min(image.width, int(x2 + w * padding))
    y2 = min(image.height, int(y2 + h * padding))
    
    return image.crop((x1, y1, x2, y2))


def extract_reduced_depth_features(depth_map, bbox):
    """
    Extracts 4 depth features: min, median, p10, bottom_center
    (NEW FUNCTION FOR MODEL 1)
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(depth_map.shape[1], x_max)
    y_max = min(depth_map.shape[0], y_max)
    
    bbox_depth = depth_map[y_min:y_max, x_min:x_max]
    
    if bbox_depth.size == 0:
        return None
    
    # 4 depth features
    min_d = float(np.min(bbox_depth))
    median_d = float(np.median(bbox_depth))
    p10 = float(np.percentile(bbox_depth, 10))
    
    # Bottom center depth
    bottom_y = min(y_max - 1, depth_map.shape[0] - 1)
    center_x = (x_min + x_max) // 2
    center_x = min(center_x, depth_map.shape[1] - 1)
    bottom_center_d = float(depth_map[bottom_y, center_x])
    
    return np.array([min_d, median_d, p10, bottom_center_d], dtype=np.float32)


def compute_geometric_distance(bbox):
    """
    Compute distance estimate using pinhole camera model
    (NEW FUNCTION FOR MODEL 1)
    """
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    
    if bbox_height <= 0:
        return 0.0
    
    distance_estimate = (AVG_CAR_HEIGHT * FOCAL_LENGTH_Y) / bbox_height
    return float(distance_estimate)


def extract_geometric_features(bbox, img_width, img_height):
    """Extract 10 geometric features for Model 2"""
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

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

    # Use hardcoded constants
    fx = CameraMat[0, 0]
    fy = CameraMat[1, 1]
    cy = CameraMat[1, 2]
    
    distance_estimate = (AVG_CAR_HEIGHT * fy) / (bbox_height + 1e-6)
    distance_estimate_norm = np.clip(distance_estimate / 100.0, 0, 1)

    vertical_angle = np.arctan2(bbox_y2 - cy, fy)
    vertical_angle_norm = (vertical_angle + np.pi/2) / np.pi

    angular_height = 2 * np.arctan(bbox_height / (2 * fy))
    angular_height_norm = angular_height / (np.pi/2)

    return torch.tensor([
        feature_1, feature_2, feature_3, feature_4, feature_5,
        feature_6, feature_7, distance_estimate_norm,
        vertical_angle_norm, angular_height_norm
    ], dtype=torch.float32)


def extract_ground_truth(filename):
    """Extract ground truth distance from filename"""
    try:
        if '_dist' in filename:
            dist_part = filename.split('_dist')[1].split('_')[0]
            return float(dist_part)
    except:
        pass
    return None


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================
def process_with_model1(image_data, detector, model1, depthpro_processor, depthpro_model, device):
    """Process with FiLMEstimator (Model 1)"""
    start_time = time.time()
    
    result = {
        'model': 'FiLMEstimator',
        'filename': image_data['filename'],
        'ground_truth': image_data['ground_truth'],
        'stages': {},
        'timings': {}
    }
    
    try:
        original = image_data['original']
        
        # Stage 1: Preprocessing (NEW: with steps)
        t0 = time.time()
        preprocessing_steps, preprocessed = preprocess_image_steps(original.copy())
        result['stages']['preprocessing_steps'] = preprocessing_steps
        result['stages']['preprocessed'] = preprocessing_steps['gamma_corrected'] # Use final step
        result['timings']['preprocessing'] = time.time() - t0
        
        # Stage 2: Detection
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Use the BGR 'processed' image for detection
            cv2.imwrite(tmp.name, preprocessed) 
            detection_result, annotated = detector.find_lead_vehicle(tmp.name, visualize=True)
            os.unlink(tmp.name)
        result['timings']['detection'] = time.time() - t0
        
        if detection_result is None or detection_result['lead_vehicle'] is None:
            result['skipped'] = True
            result['reason'] = 'No vehicle detected'
            result['total_time'] = time.time() - start_time
            return result
        
        bbox = detection_result['lead_vehicle']['bbox']
        result['stages']['detection'] = {
            'annotated': cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            'bbox': bbox,
            'confidence': detection_result['lead_vehicle']['confidence']
        }
        
        # Stage 3: Crop vehicle
        t0 = time.time()
        # Use the RGB 'gamma_corrected' image from steps for PIL
        preprocessed_pil = Image.fromarray(result['stages']['preprocessed'])
        cropped = crop_lead_vehicle(preprocessed_pil, bbox, padding=0.15)
        result['stages']['cropped'] = np.array(cropped)
        result['timings']['cropping'] = time.time() - t0
        
        # Stage 4: DepthPro extraction
        t0 = time.time()
        inputs = depthpro_processor(images=preprocessed_pil, return_tensors="pt").to(device)
        for k in inputs:
            inputs[k] = inputs[k].half()
        
        with torch.no_grad():
            outputs = depthpro_model(**inputs)
        
        depth_map = depthpro_processor.post_process_depth_estimation(
            outputs, target_sizes=[(preprocessed_pil.height, preprocessed_pil.width)]
        )[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
        result['timings']['depthpro'] = time.time() - t0
        
        # Normalize depth map for visualization
        depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
        result['stages']['depth_map'] = cv2.cvtColor(depth_vis_color, cv2.COLOR_BGR2RGB)
        
        # Extract cropped depth map
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)
        cropped_depth = depth_map[y1:y2, x1:x2]
        
        # Visualize cropped depth
        cropped_depth_vis = (cropped_depth - cropped_depth.min()) / (cropped_depth.max() - cropped_depth.min() + 1e-6)
        cropped_depth_vis = (cropped_depth_vis * 255).astype(np.uint8)
        cropped_depth_vis = cv2.applyColorMap(cropped_depth_vis, cv2.COLORMAP_VIRIDIS)
        result['stages']['cropped_depth'] = cv2.cvtColor(cropped_depth_vis, cv2.COLOR_BGR2RGB)
        
        # --- NEW 5-FEATURE EXTRACTION ---
        t0 = time.time()
        # 1. Get 4 depth features
        depth_features = extract_reduced_depth_features(depth_map, bbox)
        if depth_features is None:
            result['skipped'] = True
            result['reason'] = 'Could not extract depth features'
            result['total_time'] = time.time() - start_time
            return result
        
        # 2. Get 1 geometric feature
        geometric_distance = compute_geometric_distance(bbox)
        
        # 3. Combine into 5-feature array
        all_features = np.append(depth_features, geometric_distance).astype(np.float32)
        
        result['stages']['scalar_features'] = all_features.tolist()
        result['timings']['feature_extraction'] = time.time() - t0
        
        # Stage 5: Model prediction
        t0 = time.time()
        cropped_resized = cropped.resize((260, 260))
        cropped_tensor = transforms.ToTensor()(cropped_resized)
        cropped_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])(cropped_tensor)
        
        with torch.no_grad():
            cropped_tensor = cropped_tensor.unsqueeze(0).to(device)
            # Pass the 5-feature tensor
            scalar_tensor = torch.from_numpy(all_features).unsqueeze(0).to(device)
            prediction = model1(cropped_tensor, scalar_tensor).cpu().item()
        
        result['timings']['inference'] = time.time() - t0
        result['prediction'] = prediction
        
        if result['ground_truth'] is not None:
            result['error'] = abs(prediction - result['ground_truth'])
            result['error_percent'] = (result['error'] / result['ground_truth']) * 100
        
        result['total_time'] = time.time() - start_time
        
        # Cleanup
        del inputs, outputs, depth_map
        torch.cuda.empty_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        result['skipped'] = True
        result['reason'] = f'Error: {str(e)}'
        result['total_time'] = time.time() - start_time
        import traceback
        result['traceback'] = traceback.format_exc()
        return result


def process_with_model2(image_data, detector, model2, device):
    """Process with HybridGeometricEstimator (Model 2)"""
    start_time = time.time()
    
    result = {
        'model': 'HybridGeometricEstimator',
        'filename': image_data['filename'],
        'ground_truth': image_data['ground_truth'],
        'stages': {},
        'timings': {}
    }
    
    try:
        original = image_data['original']
        
        # Stage 1: Preprocessing (NEW: with steps)
        t0 = time.time()
        preprocessing_steps, preprocessed = preprocess_image_steps(original.copy())
        result['stages']['preprocessing_steps'] = preprocessing_steps
        result['stages']['preprocessed'] = preprocessing_steps['gamma_corrected'] # Use final step
        result['timings']['preprocessing'] = time.time() - t0
        
        # Stage 2: Detection
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, preprocessed)
            detection_result, annotated = detector.find_lead_vehicle(tmp.name, visualize=True)
            os.unlink(tmp.name)
        result['timings']['detection'] = time.time() - t0
        
        if detection_result is None or detection_result['lead_vehicle'] is None:
            result['skipped'] = True
            result['reason'] = 'No vehicle detected'
            result['total_time'] = time.time() - start_time
            return result
        
        bbox = detection_result['lead_vehicle']['bbox']
        result['stages']['detection'] = {
            'annotated': cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            'bbox': bbox,
            'confidence': detection_result['lead_vehicle']['confidence']
        }
        
        h, w = preprocessed.shape[:2]
        
        # Stage 3: Full image resize
        t0 = time.time()
        full_img = cv2.resize(preprocessed, (224, 224))
        full_img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        result['stages']['full_image'] = full_img_rgb
        
        full_img = full_img_rgb.astype(np.float32) / 255.0
        full_img = torch.from_numpy(full_img).permute(2, 0, 1)
        
        # Stage 4: Car patch extraction
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        patch = preprocessed[y1:y2, x1:x2]
        patch = cv2.resize(patch, (64, 64))
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        result['stages']['car_patch'] = patch_rgb
        
        patch = patch_rgb.astype(np.float32) / 255.0
        patch = torch.from_numpy(patch).permute(2, 0, 1)
        
        # Normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        full_img = normalize(full_img)
        patch = normalize(patch)
        
        # Stage 5: Geometric features
        geometric = extract_geometric_features(bbox, w, h)
        result['stages']['geometric_features'] = geometric.numpy().tolist()
        result['timings']['feature_extraction'] = time.time() - t0
        
        # Stage 6: Model prediction
        t0 = time.time()
        with torch.no_grad():
            full_img = full_img.unsqueeze(0).to(device)
            patch = patch.unsqueeze(0).to(device)
            geometric = geometric.unsqueeze(0).to(device)
            
            prediction = model2(full_img, patch, geometric).cpu().item()
        
        result['timings']['inference'] = time.time() - t0
        result['prediction'] = prediction
        
        if result['ground_truth'] is not None:
            result['error'] = abs(prediction - result['ground_truth'])
            result['error_percent'] = (result['error'] / result['ground_truth']) * 100
        
        result['total_time'] = time.time() - start_time
        
        return result
        
    except Exception as e:
        result['skipped'] = True
        result['reason'] = f'Error: {str(e)}'
        result['total_time'] = time.time() - start_time
        import traceback
        result['traceback'] = traceback.format_exc()
        return result


# ============================================================================
# MAIN STREAMLIT UI
# ============================================================================
def main():
    st.set_page_config(
        page_title="Dual Model Distance Estimation",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # IMPROVED DARK MODE SUPPORT WITH HIGH CONTRAST
    st.markdown("""
    <style>
    /* Dark mode improvements */
    .main {
        background-color: #0e1117;
    }
    
    /* High contrast text for dark mode */
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border: 1px solid #3a3a4a;
        border-radius: 5px;
    }
    
    .stMetric label {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #fafafa !important;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    /* Text elements */
    p, span, div {
        color: #e0e0e0 !important;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #1e1e1e;
    }
    
    .stDataFrame td, .stDataFrame th {
        color: #ffffff !important;
        background-color: #262730 !important;
    }
    
    /* Captions */
    .caption {
        color: #b0b0b0 !important;
        font-size: 0.9rem;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1e3a5f;
        color: #ffffff;
        border: 1px solid #2a5080;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #1e4d2b;
        color: #ffffff;
        border: 1px solid #2d6b3d;
    }
    
    /* Feature value display */
    .feature-box {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border: 1px solid #3a3a4a;
    }
    
    .feature-name {
        color: #a0a0a0 !important;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .feature-value {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0e1117;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    
    /* Buttons */
    .stButton button {
        color: #ffffff;
        border: 1px solid #3a3a4a;
    }
    
    /* Input fields */
    .stTextInput input {
        color: #ffffff !important;
        background-color: #262730 !important;
        border: 1px solid #3a3a4a !important;
    }
    
    /* Markdown text */
    .markdown-text-container {
        color: #e0e0e0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Dual Model Distance Estimation Pipeline")
    st.markdown("**Compare Model 1 (FiLMEstimator) vs Model 2 (Hybrid Geometric Estimator)**")

    # Initialize session state
    if 'images_loaded' not in st.session_state:
        st.session_state.images_loaded = []
    if 'results_model1' not in st.session_state:
        st.session_state.results_model1 = []
    if 'results_model2' not in st.session_state:
        st.session_state.results_model2 = []

    # SIDEBAR CONFIGURATION
    st.sidebar.header("Configuration")

    st.sidebar.subheader("Model Selection")
    # [!!! CHANGE !!!]
    run_model1 = st.sidebar.checkbox("Run Model 1: FiLMEstimator (5-Feature)", value=True)
    run_model2 = st.sidebar.checkbox("Run Model 2: Hybrid Geometric Estimator", value=True)

    st.sidebar.markdown("---")
    
    # MODEL PATHS
    st.sidebar.subheader("Model Paths")
    model1_path = st.sidebar.text_input(
        "Model 1 Path (New 5-Feature)", 
        "best_model.pth" # Default name from new train script
    )
    model2_path = st.sidebar.text_input(
        "Model 2 Path (Old 3-Branch)", 
        "best_model_hybrid.pt" # Renamed to avoid conflict
    )
    
    st.sidebar.markdown("---")
    
    conf_threshold = st.sidebar.slider("YOLO Confidence", 0.10, 0.50, 0.25, 0.05)

    st.sidebar.markdown("---")
    
    # Image upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload Images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

    # Load images
    if st.sidebar.button("Load Images", use_container_width=True) and uploaded_files:
        st.session_state.images_loaded = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            st.session_state.images_loaded.append({
                'filename': file.name,
                'original': image,
                'ground_truth': extract_ground_truth(file.name)
            })

            progress_bar.progress((idx + 1) / len(uploaded_files))
            status_text.text(f"Loading {idx + 1}/{len(uploaded_files)}")

        progress_bar.empty()
        status_text.success(f"Loaded {len(uploaded_files)} images")

    st.sidebar.markdown("---")

    # RUN PIPELINE
    if st.sidebar.button("Run Pipeline", use_container_width=True, type="primary"):
        
        if len(st.session_state.images_loaded) == 0:
            st.error("No images loaded!")
            return
        
        if not run_model1 and not run_model2:
            st.error("Select at least one model!")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Device: {device}")

        # Load models
        with st.spinner("Loading models..."):
            try:
                detector = LeadVehicleDetector(
                    model_path='yolov8m.pt',
                    conf_threshold=conf_threshold,
                    use_adaptive_roi=True,
                    device=device
                )
                
                # Load Model 1
                model1, depthpro_processor, depthpro_model = None, None, None
                if run_model1:
                    if not os.path.exists(model1_path):
                        st.error(f"Model 1 not found: {model1_path}")
                        return
                    
                    # [!!! CHANGE !!!] LOAD NEW MODEL
                    model1 = FiLMEstimator(backbone="efficientnet_b2", pretrained=False)
                    checkpoint = torch.load(model1_path, map_location=device)
                    model1.load_state_dict(checkpoint['model_state_dict'])
                    model1 = model1.to(device).eval()
                    
                    # Load DepthPro
                    st.text("Loading DepthPro model (this may take 10-20 seconds)...")
                    depthpro_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
                    depthpro_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf")
                    depthpro_model = depthpro_model.to(device).half().eval()
                    st.success("Model 1 + DepthPro loaded")
                
                # Load Model 2
                model2 = None
                if run_model2:
                    if not os.path.exists(model2_path):
                        st.error(f"Model 2 not found: {model2_path}")
                        return
                    
                    model2 = HybridGeometricEstimator()
                    checkpoint = torch.load(model2_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        model2.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model2.load_state_dict(checkpoint)
                    model2 = model2.to(device).eval()
                    st.success("Model 2 loaded")
                
            except Exception as e:
                st.error(f"Error loading models: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        # Process images
        st.session_state.results_model1 = []
        st.session_state.results_model2 = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, img_data in enumerate(st.session_state.images_loaded):
            status_text.text(f"Processing {idx + 1}/{len(st.session_state.images_loaded)}: {img_data['filename']}")
            
            if run_model1:
                result1 = process_with_model1(img_data, detector, model1, depthpro_processor, depthpro_model, device)
                st.session_state.results_model1.append(result1)
            
            if run_model2:
                result2 = process_with_model2(img_data, detector, model2, device)
                st.session_state.results_model2.append(result2)
            
            progress_bar.progress((idx + 1) / len(st.session_state.images_loaded))

        progress_bar.empty()
        status_text.success("Pipeline complete!")

    # DISPLAY RESULTS
    if len(st.session_state.results_model1) > 0 or len(st.session_state.results_model2) > 0:
        
        st.markdown("---")
        st.header("Results")

        # Image selector
        all_results = st.session_state.results_model1 if len(st.session_state.results_model1) > 0 else st.session_state.results_model2
        image_names = [r['filename'] for r in all_results]
        
        selected_idx = st.selectbox(
            "Select Image",
            range(len(image_names)),
            format_func=lambda x: f"{x+1}. {image_names[x]}"
        )

        # Get results for both models
        result1 = st.session_state.results_model1[selected_idx] if len(st.session_state.results_model1) > 0 else None
        result2 = st.session_state.results_model2[selected_idx] if len(st.session_state.results_model2) > 0 else None

        # SIDE-BY-SIDE COMPARISON
        if result1 and result2:
            st.subheader("Side-by-Side Comparison")
            
            col_left, col_right = st.columns(2)
            
            # LEFT: MODEL 1
            with col_left:
                # [!!! CHANGE !!!]
                st.markdown("### Model 1: FiLMEstimator")
                
                if result1.get('skipped'):
                    st.warning(f"Skipped: {result1['reason']}")
                else:
                    # Timing info
                    st.caption(f"Total: {result1['total_time']:.3f}s | Inference: {result1['timings']['inference']:.3f}s")
                    
                    # Preprocessing (NEW: 4 steps)
                    st.markdown("**Stage 1: Preprocessing**")
                    prepro_cols = st.columns(4)
                    prepro_cols[0].image(result1['stages']['preprocessing_steps']['undistorted'], caption="Undistorted", use_container_width=True)
                    prepro_cols[1].image(result1['stages']['preprocessing_steps']['white_balanced'], caption="White Balanced", use_container_width=True)
                    prepro_cols[2].image(result1['stages']['preprocessing_steps']['clahe_applied'], caption="CLAHE", use_container_width=True)
                    prepro_cols[3].image(result1['stages']['preprocessed'], caption="Gamma", use_container_width=True)
                    
                    # Detection
                    st.markdown("**Stage 2: Detection**")
                    st.image(result1['stages']['detection']['annotated'], caption="Detected Vehicle", use_container_width=True)
                    st.caption(f"Confidence: {result1['stages']['detection']['confidence']:.3f}")
                    
                    # Cropped vehicle
                    st.markdown("**Stage 3: Cropped Vehicle**")
                    st.image(result1['stages']['cropped'], caption="Cropped Vehicle (with padding)", use_container_width=True)
                    
                    # Depth maps
                    st.markdown("**Stage 4: DepthPro Analysis**")
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.image(result1['stages']['depth_map'], caption="Full Depth Map", use_container_width=True)
                    with dcol2:
                        st.image(result1['stages']['cropped_depth'], caption="Cropped Depth", use_container_width=True)
                    
                    # 5 Scalar Features - HIGH CONTRAST
                    st.markdown("**5 Scalar Features:**")
                    feature_names = ["Min Depth", "Median", "P10", "Bottom", "Geo. Est."]
                    fcols = st.columns(5)
                    for i, (name, val) in enumerate(zip(feature_names, result1['stages']['scalar_features'])):
                        with fcols[i % 5]:
                            st.markdown(f"""
                            <div class="feature-box">
                                <div class="feature-name">{name}</div>
                                <div class="feature-value">{val:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Prediction
                    st.markdown("**Stage 5: Prediction**")
                    pcol1, pcol2, pcol3 = st.columns(3)
                    with pcol1:
                        gt = result1['ground_truth']
                        st.metric("Ground Truth", f"{gt:.2f}m" if gt else "N/A")
                    with pcol2:
                        st.metric("Predicted", f"{result1['prediction']:.2f}m")
                    with pcol3:
                        if result1.get('error'):
                            st.metric("Error", f"{result1['error']:.2f}m", 
                                      delta=f"{result1['error_percent']:.1f}%", delta_color="inverse")
            
            # RIGHT: MODEL 2
            with col_right:
                st.markdown("### Model 2: Hybrid Geometric Estimator")
                
                if result2.get('skipped'):
                    st.warning(f"Skipped: {result2['reason']}")
                else:
                    # Timing info
                    st.caption(f"Total: {result2['total_time']:.3f}s | Inference: {result2['timings']['inference']:.3f}s")
                    
                    # Preprocessing (NEW: 4 steps)
                    st.markdown("**Stage 1: Preprocessing**")
                    prepro_cols = st.columns(4)
                    prepro_cols[0].image(result2['stages']['preprocessing_steps']['undistorted'], caption="Undistorted", use_container_width=True)
                    prepro_cols[1].image(result2['stages']['preprocessing_steps']['white_balanced'], caption="White Balanced", use_container_width=True)
                    prepro_cols[2].image(result2['stages']['preprocessing_steps']['clahe_applied'], caption="CLAHE", use_container_width=True)
                    prepro_cols[3].image(result2['stages']['preprocessed'], caption="Gamma", use_container_width=True)
                    
                    # Detection
                    st.markdown("**Stage 2: Detection**")
                    st.image(result2['stages']['detection']['annotated'], caption="Detected Vehicle", use_container_width=True)
                    st.caption(f"Confidence: {result2['stages']['detection']['confidence']:.3f}")
                    
                    # Full image and patch
                    st.markdown("**Stage 3: Image Inputs**")
                    icol1, icol2 = st.columns(2)
                    with icol1:
                        st.image(result2['stages']['full_image'], caption="Full Image (224x224)", use_container_width=True)
                    with icol2:
                        st.image(result2['stages']['car_patch'], caption="Car Patch (64x64)", use_container_width=True)
                    
                    # Geometric features - HIGH CONTRAST
                    st.markdown("**Stage 4: 10 Geometric Features:**")
                    geo_names = [
                        "Bbox W", "Bbox H", "Y1", "Y2", "Center X",
                        "Center Y", "Aspect", "Dist Est", "V Angle", "Angular H"
                    ]
                    for i in range(0, len(geo_names), 2):
                        gcols = st.columns(2)
                        for j in range(2):
                            if i + j < len(geo_names):
                                with gcols[j]:
                                    name = geo_names[i + j]
                                    val = result2['stages']['geometric_features'][i + j]
                                    st.markdown(f"""
                                    <div class="feature-box">
                                        <div class="feature-name">{name}</div>
                                        <div class="feature-value">{val:.3f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Prediction
                    st.markdown("**Stage 5: Prediction**")
                    pcol1, pcol2, pcol3 = st.columns(3)
                    with pcol1:
                        gt = result2['ground_truth']
                        st.metric("Ground Truth", f"{gt:.2f}m" if gt else "N/A")
                    with pcol2:
                        st.metric("Predicted", f"{result2['prediction']:.2f}m")
                    with pcol3:
                        if result2.get('error'):
                            st.metric("Error", f"{result2['error']:.2f}m", 
                                      delta=f"{result2['error_percent']:.1f}%", delta_color="inverse")

        # SINGLE MODEL VIEW
        elif result1:
            # [!!! CHANGE !!!]
            st.subheader("Model 1: FiLMEstimator Results")
            
            if result1.get('skipped'):
                st.warning(f"Skipped: {result1['reason']}")
                if result1.get('traceback'):
                    with st.expander("Error Details"):
                        st.code(result1['traceback'])
            else:
                # Timing summary
                st.info(f"Total: {result1['total_time']:.3f}s | "
                        f"Preprocessing: {result1['timings']['preprocessing']:.3f}s | "
                        f"Detection: {result1['timings']['detection']:.3f}s | "
                        f"DepthPro: {result1['timings']['depthpro']:.3f}s | "
                        f"Inference: {result1['timings']['inference']:.3f}s")
                
                # Full pipeline display
                st.markdown("**Full Pipeline**")
                cols = st.columns(7)
                original_rgb = cv2.cvtColor(st.session_state.images_loaded[selected_idx]['original'], cv2.COLOR_BGR2RGB)
                cols[0].image(original_rgb, caption="Original", use_container_width=True)
                # Show all steps
                cols[1].image(result1['stages']['preprocessing_steps']['undistorted'], caption="Undistorted", use_container_width=True)
                cols[2].image(result1['stages']['preprocessing_steps']['white_balanced'], caption="WB", use_container_width=True)
                cols[3].image(result1['stages']['preprocessing_steps']['clahe_applied'], caption="CLAHE", use_container_width=True)
                cols[4].image(result1['stages']['preprocessed'], caption="Gamma", use_container_width=True)
                cols[5].image(result1['stages']['detection']['annotated'], caption="Detected", use_container_width=True)
                cols[6].image(result1['stages']['cropped'], caption="Cropped", use_container_width=True)
                
                st.markdown("---")
                
                # Features and prediction
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**5 Scalar Features:**")
                    feature_names = ["Min Depth", "Median", "P10", "Bottom", "Geo. Est."]
                    for name, val in zip(feature_names, result1['stages']['scalar_features']):
                        st.markdown(f"""
                        <div class="feature-box">
                            <span class="feature-name">{name}:</span>
                            <span class="feature-value">{val:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Prediction:**")
                    st.metric("Ground Truth", f"{result1['ground_truth']:.2f}m" if result1['ground_truth'] else "N/A")
                    st.metric("Predicted Distance", f"{result1['prediction']:.2f}m")
                    if result1.get('error'):
                        st.metric("Error", f"{result1['error']:.2f}m ({result1['error_percent']:.1f}%)", delta_color="inverse")

        elif result2:
            st.subheader("Model 2: Hybrid Geometric Estimator Results")
            
            if result2.get('skipped'):
                st.warning(f"Skipped: {result2['reason']}")
                if result2.get('traceback'):
                    with st.expander("Error Details"):
                        st.code(result2['traceback'])
            else:
                # Timing summary
                st.info(f"Total: {result2['total_time']:.3f}s | "
                        f"Preprocessing: {result2['timings']['preprocessing']:.3f}s | "
                        f"Detection: {result2['timings']['detection']:.3f}s | "
                        f"Feature Extraction: {result2['timings']['feature_extraction']:.3f}s | "
                        f"Inference: {result2['timings']['inference']:.3f}s")
                
                # Full pipeline display
                st.markdown("**Full Pipeline**")
                cols = st.columns(7)
                original_rgb = cv2.cvtColor(st.session_state.images_loaded[selected_idx]['original'], cv2.COLOR_BGR2RGB)
                cols[0].image(original_rgb, caption="Original", use_container_width=True)
                cols[1].image(result2['stages']['preprocessing_steps']['undistorted'], caption="Undistorted", use_container_width=True)
                cols[2].image(result2['stages']['preprocessing_steps']['white_balanced'], caption="WB", use_container_width=True)
                cols[3].image(result2['stages']['preprocessing_steps']['clahe_applied'], caption="CLAHE", use_container_width=True)
                cols[4].image(result2['stages']['preprocessed'], caption="Gamma", use_container_width=True)
                cols[5].image(result2['stages']['detection']['annotated'], caption="Detected", use_container_width=True)
                cols[6].image(result2['stages']['full_image'], caption="Full (224x224)", use_container_width=True)
                # Note: car_patch is too small to be useful here, so 7 columns is fine.
                
                st.markdown("---")
                
                # Features and prediction
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**10 Geometric Features:**")
                    geo_names = [
                        "Bbox Width", "Bbox Height", "Y1", "Y2", "Center X",
                        "Center Y", "Aspect Ratio", "Distance Est", "Vertical Angle", "Angular Height"
                    ]
                    for name, val in zip(geo_names, result2['stages']['geometric_features']):
                        st.markdown(f"""
                        <div class="feature-box">
                            <span class="feature-name">{name}:</span>
                            <span class="feature-value">{val:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Prediction:**")
                    st.metric("Ground Truth", f"{result2['ground_truth']:.2f}m" if result2['ground_truth'] else "N/A")
                    st.metric("Predicted Distance", f"{result2['prediction']:.2f}m")
                    if result2.get('error'):
                        st.metric("Error", f"{result2['error']:.2f}m ({result2['error_percent']:.1f}%)", delta_color="inverse")

        # BATCH ANALYTICS
        st.markdown("---")
        st.header("Batch Analytics")

        # Filter valid results
        valid_results1 = [r for r in st.session_state.results_model1 
                          if not r.get('skipped') and r.get('ground_truth')]
        valid_results2 = [r for r in st.session_state.results_model2 
                          if not r.get('skipped') and r.get('ground_truth')]

        if len(valid_results1) > 0 or len(valid_results2) > 0:
            
            cols = st.columns(3)
            
            # Model 1 stats
            if len(valid_results1) > 0:
                errors1 = [r['error'] for r in valid_results1]
                mae1 = np.mean(errors1)
                rmse1 = np.sqrt(np.mean([e**2 for e in errors1]))
                avg_time1 = np.mean([r['total_time'] for r in valid_results1])
                avg_inference1 = np.mean([r['timings']['inference'] for r in valid_results1])
                
                with cols[0]:
                    # [!!! CHANGE !!!]
                    st.markdown("**Model 1: FiLMEstimator**")
                    st.metric("MAE", f"{mae1:.3f}m")
                    st.metric("RMSE", f"{rmse1:.3f}m")
                    st.metric("Avg Total Time", f"{avg_time1:.3f}s")
                    st.metric("Avg Inference Time", f"{avg_inference1:.3f}s")
                    st.metric("Valid", f"{len(valid_results1)}/{len(st.session_state.results_model1)}")
            
            # Model 2 stats
            if len(valid_results2) > 0:
                errors2 = [r['error'] for r in valid_results2]
                mae2 = np.mean(errors2)
                rmse2 = np.sqrt(np.mean([e**2 for e in errors2]))
                avg_time2 = np.mean([r['total_time'] for r in valid_results2])
                avg_inference2 = np.mean([r['timings']['inference'] for r in valid_results2])
                
                with cols[1]:
                    st.markdown("**Model 2: Hybrid Geometric**")
                    st.metric("MAE", f"{mae2:.3f}m")
                    st.metric("RMSE", f"{rmse2:.3f}m")
                    st.metric("Avg Total Time", f"{avg_time2:.3f}s")
                    st.metric("Avg Inference Time", f"{avg_inference2:.3f}s")
                    st.metric("Valid", f"{len(valid_results2)}/{len(st.session_state.results_model2)}")
            
            # Comparison
            if len(valid_results1) > 0 and len(valid_results2) > 0:
                with cols[2]:
                    st.markdown("**Winner**")
                    if mae1 < mae2:
                        # [!!! CHANGE !!!]
                        st.success("Model 1 (FiLMEstimator)")
                        improvement = ((mae2 - mae1) / mae2) * 100
                        st.metric("Accuracy Improvement", f"{improvement:.1f}%")
                    else:
                        st.success("Model 2 (Hybrid Geometric)")
                        improvement = ((mae1 - mae2) / mae1) * 100
                        st.metric("Accuracy Improvement", f"{improvement:.1f}%")
                    
                    # Speed comparison
                    if avg_time1 < avg_time2:
                        speedup = ((avg_time2 - avg_time1) / avg_time2) * 100
                        st.metric("Speed Advantage", f"Model 1 ({speedup:.1f}% faster)")
                    else:
                        speedup = ((avg_time1 - avg_time2) / avg_time1) * 100
                        st.metric("Speed Advantage", f"Model 2 ({speedup:.1f}% faster)")

        # RESULTS TABLE
        st.markdown("---")
        st.subheader("All Results")

        table_data = []
        
        for idx in range(len(all_results)):
            r1 = st.session_state.results_model1[idx] if len(st.session_state.results_model1) > 0 else None
            r2 = st.session_state.results_model2[idx] if len(st.session_state.results_model2) > 0 else None
            
            filename = r1['filename'] if r1 else r2['filename']
            gt = r1['ground_truth'] if r1 else r2['ground_truth']
            
            row = {'Filename': filename, 'Ground Truth (m)': f"{gt:.2f}" if gt else "N/A"}
            
            if r1:
                if r1.get('skipped'):
                    row['Model 1 Pred'] = "Skipped"
                    row['Model 1 Error'] = "N/A"
                    row['Model 1 Time'] = f"{r1.get('total_time', 0):.3f}s"
                else:
                    row['Model 1 Pred'] = f"{r1['prediction']:.2f}m"
                    row['Model 1 Error'] = f"{r1['error']:.2f}m" if r1.get('error') else "N/A"
                    row['Model 1 Time'] = f"{r1['total_time']:.3f}s"
            
            if r2:
                if r2.get('skipped'):
                    row['Model 2 Pred'] = "Skipped"
                    row['Model 2 Error'] = "N/A"
                    row['Model 2 Time'] = f"{r2.get('total_time', 0):.3f}s"
                else:
                    row['Model 2 Pred'] = f"{r2['prediction']:.2f}m"
                    row['Model 2 Error'] = f"{r2['error']:.2f}m" if r2.get('error') else "N/A"
                    row['Model 2 Time'] = f"{r2['total_time']:.3f}s"
            
            table_data.append(row)

        st.dataframe(table_data, use_container_width=True, height=300)

        # EXPORT
        st.markdown("---")
        
        if st.button("Export Results (JSON)", use_container_width=False):
            export_data = {
                'model1_results': st.session_state.results_model1,
                'model2_results': st.session_state.results_model2,
                'summary': {
                    'model1_valid': len(valid_results1) if len(valid_results1) > 0 else 0,
                    'model2_valid': len(valid_results2) if len(valid_results2) > 0 else 0,
                    'model1_mae': float(np.mean([r['error'] for r in valid_results1])) if len(valid_results1) > 0 else None,
                    'model2_mae': float(np.mean([r['error'] for r in valid_results2])) if len(valid_results2) > 0 else None,
                    'model1_avg_time': float(np.mean([r['total_time'] for r in valid_results1])) if len(valid_results1) > 0 else None,
                    'model2_avg_time': float(np.mean([r['total_time'] for r in valid_results2])) if len(valid_results2) > 0 else None,
                }
            }
            
            # Use a helper function for JSON serialization to avoid numpy errors
            def convert(o):
                if isinstance(o, np.generic): return o.item()
                raise TypeError

            json_str = json.dumps(export_data, indent=2, default=convert)
            st.download_button(
                label="Download JSON File",
                data=json_str,
                file_name="dual_model_comparison.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
