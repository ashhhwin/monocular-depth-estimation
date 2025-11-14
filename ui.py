"""
Dashcam Distance Estimation Pipeline - Streamlit UI
Final production inference script with Git integration

RUN THIS IN COLAB/JUPYTER:
    !git clone https://github.com/ashhhwin/monocular-depth-estimation.git
    %cd monocular-depth-estimation
    !pip install streamlit torch torchvision opencv-python ultralytics numpy
    !streamlit run ui.py --server.port 8501
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import json
import tempfile
import sys

import warnings
warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
st.set_option('client.showErrorDetails', False)

# Add the cloned repo to Python path
REPO_PATH = os.getcwd()  # current working directory
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Now import from the cloned repo
try:
    from lead_vehicle_detector import LeadVehicleDetector
    from preprocessing import (undistort_image, apply_white_balance, 
                              apply_clahe, apply_gamma_correction)
    import config
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error(f"Current directory: {os.getcwd()}")
    st.error(f"Files in directory: {os.listdir('.')}")
    st.stop()


class HybridDistanceEstimator(nn.Module):
    """3-branch model: EfficientNetB4 + MobileNetV2 + Geometric Features"""
    
    def __init__(self):
        super(HybridDistanceEstimator, self).__init__()
        
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


def extract_geometric_features(bbox, img_width, img_height):
    """Extract 10 geometric features from bounding box"""
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
    
    fx = 2429.865965
    fy = 2424.492001
    cx = 1192.584876
    cy = 1015.978074
    AVG_CAR_HEIGHT = 1.5
    
    distance_estimate = (AVG_CAR_HEIGHT * fy) / (bbox_height + 1e-6)
    distance_estimate_norm = np.clip(distance_estimate / 100.0, 0, 1)
    
    vertical_angle = np.arctan2(bbox_y2 - cy, fy)
    vertical_angle_norm = (vertical_angle + np.pi/2) / np.pi
    
    angular_height = 2 * np.arctan(bbox_height / (2 * fy))
    angular_height_norm = angular_height / (np.pi/2)
    
    geometric_features = torch.tensor([
        feature_1, feature_2, feature_3, feature_4, feature_5,
        feature_6, feature_7, distance_estimate_norm,
        vertical_angle_norm, angular_height_norm
    ], dtype=torch.float32)
    
    return geometric_features


def prepare_model_inputs(image, bbox):
    """Prepare full image, car patch, and geometric features for model"""
    h, w = image.shape[:2]
    
    # Full image resize to 224x224
    full_img = cv2.resize(image, (224, 224))
    full_img = full_img.astype(np.float32) / 255.0
    full_img = torch.from_numpy(full_img).permute(2, 0, 1)
    
    # Car patch extraction and resize to 64x64
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    patch = image[y1:y2, x1:x2]
    patch = cv2.resize(patch, (64, 64))
    patch = patch.astype(np.float32) / 255.0
    patch = torch.from_numpy(patch).permute(2, 0, 1)
    
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    full_img = normalize(full_img)
    patch = normalize(patch)
    
    # Geometric features
    geometric = extract_geometric_features(bbox, w, h)
    
    return full_img.unsqueeze(0), patch.unsqueeze(0), geometric.unsqueeze(0)


def extract_ground_truth(filename):
    """Extract ground truth distance from filename pattern: seq*_dist*.jpg"""
    try:
        if '_dist' in filename:
            dist_part = filename.split('_dist')[1].split('_')[0]
            return float(dist_part)
    except:
        pass
    return None


def process_single_image(image_data, detector, model, device):
    """Process a single image through the complete pipeline"""
    
    # Create temporary file for detector
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image_data['original'])
    
    results = {
        'filename': image_data['filename'],
        'ground_truth': image_data['ground_truth'],
        'preprocessing': {},
        'detection': None,
        'features': None,
        'prediction': None,
        'error': None,
        'error_percent': None
    }
    
    try:
        # STAGE 1: PREPROCESSING
        original = image_data['original']
        undistorted = undistort_image(original.copy())
        white_balanced = apply_white_balance(undistorted.copy())
        clahe_applied = apply_clahe(white_balanced.copy())
        gamma_corrected = apply_gamma_correction(clahe_applied.copy())
        
        results['preprocessing'] = {
            'original': cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            'undistorted': cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB),
            'white_balanced': cv2.cvtColor(white_balanced, cv2.COLOR_BGR2RGB),
            'clahe': cv2.cvtColor(clahe_applied, cv2.COLOR_BGR2RGB),
            'gamma_corrected': cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
        }
        
        # STAGE 2: VEHICLE DETECTION
        detection_result, annotated = detector.find_lead_vehicle(tmp_path, visualize=True)
        
        if detection_result is None or detection_result['lead_vehicle'] is None:
            results['skipped'] = True
            results['reason'] = 'No lead vehicle detected'
            os.unlink(tmp_path)
            return results
        
        results['detection'] = {
            'bbox': detection_result['lead_vehicle']['bbox'],
            'confidence': detection_result['lead_vehicle']['confidence'],
            'in_roi': detection_result['lead_vehicle']['in_roi'],
            'annotated': cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        }
        
        # STAGE 3: FEATURE EXTRACTION
        bbox = detection_result['lead_vehicle']['bbox']
        full_img, car_patch, geometric = prepare_model_inputs(gamma_corrected, bbox)
        
        results['features'] = {
            'geometric_values': geometric.squeeze().numpy().tolist()
        }
        
        # STAGE 4: DISTANCE PREDICTION
        with torch.no_grad():
            full_img = full_img.to(device)
            car_patch = car_patch.to(device)
            geometric = geometric.to(device)
            
            prediction = model(full_img, car_patch, geometric)
            prediction = prediction.cpu().item()
        
        results['prediction'] = prediction
        
        # Calculate error if ground truth exists
        if results['ground_truth'] is not None:
            results['error'] = abs(prediction - results['ground_truth'])
            results['error_percent'] = (results['error'] / results['ground_truth']) * 100
        
        os.unlink(tmp_path)
        return results
        
    except Exception as e:
        results['skipped'] = True
        results['reason'] = f'Error: {str(e)}'
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return results


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Dashcam Distance Estimation",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    /* Adapt to dark/light mode */
    html, body, .main {
        background-color: transparent !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0e1117; /* Dark background */
        color: #e0e0e0;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #e0e0e0 !important;
    }

    .stMetric {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 15px;
    }

    .stButton>button {
        background-color: #262730 !important;
        color: #ffffff !important;
        border: 1px solid #444;
        border-radius: 8px;
    }

    .stButton>button:hover {
        background-color: #444 !important;
    }

    .stDataFrame, .stTable {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
    }

    code {
        color: #d0d0d0 !important;
        background-color: #222 !important;
        border-radius: 5px;
        padding: 3px 6px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("Dashcam Distance Estimation Pipeline")
    st.markdown("**Multi-stage processing:** Preprocessing → Detection → Feature Extraction → Distance Prediction")
    
    # Show current directory info
    with st.expander("System Information"):
        st.code(f"Working Directory: {os.getcwd()}")
        st.code(f"Python Path: {sys.path[:3]}")
        st.code(f"Files Available: {', '.join(os.listdir('.')[:10])}")
    
    # Initialize session state
    if 'images_loaded' not in st.session_state:
        st.session_state.images_loaded = []
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # SIDEBAR CONFIGURATION
    st.sidebar.header("Configuration")
    
    model_path = st.sidebar.text_input("Distance Model Path", "best_model.pt")
    conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 0.50, 0.25, 0.05)
    
    st.sidebar.markdown("---")
    
    # Image upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload Images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Select one or multiple dashcam images"
    )
    
    # Load images button
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
            status_text.text(f"Loading {idx + 1}/{len(uploaded_files)} images")
        
        progress_bar.empty()
        status_text.success(f"✓ Loaded {len(uploaded_files)} images")
    
    st.sidebar.markdown("---")
    
    # Run pipeline button
    if st.sidebar.button("Run Pipeline", use_container_width=True, type="primary"):
        
        if len(st.session_state.images_loaded) == 0:
            st.error("No images loaded. Please upload and load images first.")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Using device: {device}")
        
        # Load models
        with st.spinner("Loading models..."):
            try:
                # Load distance estimation model
                model = HybridDistanceEstimator()
                
                # Check if model file exists
                if not os.path.exists(model_path):
                    st.error(f"Model file not found: {model_path}")
                    st.info("Please ensure best_model.pt is in the same directory as this script")
                    return
                
                checkpoint = torch.load(model_path, map_location=device)
                #model.load_state_dict(torch.load(model_path, map_location=device))
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()
                st.session_state.model = model
                
                # Load vehicle detector
                detector = LeadVehicleDetector(
                    model_path='yolov8m.pt',
                    conf_threshold=conf_threshold,
                    use_adaptive_roi=True,
                    device=device
                )
                st.session_state.detector = detector
                
                st.success("✓ Models loaded successfully")
                
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
        
        # Process images
        st.session_state.results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, img_data in enumerate(st.session_state.images_loaded):
            status_text.text(f"Processing {idx + 1}/{len(st.session_state.images_loaded)}: {img_data['filename']}")
            
            result = process_single_image(
                img_data, 
                st.session_state.detector, 
                st.session_state.model, 
                device
            )
            st.session_state.results.append(result)
            
            progress_bar.progress((idx + 1) / len(st.session_state.images_loaded))
        
        progress_bar.empty()
        status_text.success("✓ Pipeline complete")
    
    # RESULTS DISPLAY
    if len(st.session_state.results) > 0:
        
        st.markdown("---")
        st.header("Results")
        
        # Image selector
        image_names = [r['filename'] for r in st.session_state.results]
        selected_idx = st.selectbox(
            "Select Image",
            range(len(image_names)),
            format_func=lambda x: f"{x+1}. {image_names[x]}"
        )
        
        result = st.session_state.results[selected_idx]
        
        # Check if skipped
        if result.get('skipped'):
            st.warning(f"⚠ {result['reason']}")
        else:
            
            # STAGE 1: PREPROCESSING
            st.subheader("Stage 1: Preprocessing Pipeline")
            cols = st.columns(5)
            steps = ['original', 'undistorted', 'white_balanced', 'clahe', 'gamma_corrected']
            step_labels = ['Original', 'Undistorted', 'White Balanced', 'CLAHE', 'Gamma Corrected']
            
            for col, step, label in zip(cols, steps, step_labels):
                with col:
                    st.image(
                        result['preprocessing'][step],
                        caption=label,
                        use_column_width=True
                    )
            
            st.markdown("---")
            
            # STAGE 2: DETECTION
            st.subheader("Stage 2: Lead Vehicle Detection")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(
                    result['detection']['annotated'],
                    caption="Detection Result",
                    use_column_width=True
                )
            
            with col2:
                st.metric("Confidence", f"{result['detection']['confidence']:.3f}")
                st.metric("In ROI", "Yes" if result['detection']['in_roi'] else "No")
                
                st.text("Bounding Box Coordinates:")
                bbox = result['detection']['bbox']
                st.code(
                    f"X1: {bbox[0]:.1f}\n"
                    f"Y1: {bbox[1]:.1f}\n"
                    f"X2: {bbox[2]:.1f}\n"
                    f"Y2: {bbox[3]:.1f}",
                    language=None
                )
            
            st.markdown("---")
            
            # STAGE 3: FEATURES
            st.subheader("Stage 3: Geometric Feature Extraction")
            
            feature_names = [
                "Bbox Width (norm)", "Bbox Height (norm)", "Bbox Y1 (norm)",
                "Bbox Y2 (norm)", "Center X (norm)", "Center Y (norm)",
                "Aspect Ratio", "Distance Est. (norm)",
                "Vertical Angle (norm)", "Angular Height (norm)"
            ]
            
            cols = st.columns(5)
            for idx, (name, val) in enumerate(zip(feature_names, result['features']['geometric_values'])):
                with cols[idx % 5]:
                    st.metric(name, f"{val:.4f}")
            
            st.markdown("---")
            
            # STAGE 4: PREDICTION
            st.subheader("Stage 4: Distance Estimation")
            
            cols = st.columns(3)
            
            with cols[0]:
                gt = result['ground_truth']
                st.metric(
                    "Ground Truth",
                    f"{gt:.2f} m" if gt is not None else "N/A"
                )
            
            with cols[1]:
                st.metric("Predicted Distance", f"{result['prediction']:.2f} m")
            
            with cols[2]:
                if result['error'] is not None:
                    st.metric(
                        "Error",
                        f"±{result['error']:.2f} m",
                        delta=f"{result['error_percent']:.1f}%"
                    )
        
        # BATCH SUMMARY
        st.markdown("---")
        st.header("Batch Analytics")
        
        valid_results = [
            r for r in st.session_state.results
            if not r.get('skipped') and r['ground_truth'] is not None
        ]
        
        if len(valid_results) > 0:
            errors = [r['error'] for r in valid_results]
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Mean Absolute Error (MAE)", f"{mae:.3f} m")
            
            with cols[1]:
                st.metric("Root Mean Square Error (RMSE)", f"{rmse:.3f} m")
            
            with cols[2]:
                st.metric("Valid Predictions", f"{len(valid_results)}/{len(st.session_state.results)}")
        
        # RESULTS TABLE
        st.subheader("All Results")
        
        table_data = []
        for r in st.session_state.results:
            table_data.append({
                'Filename': r['filename'],
                'Ground Truth (m)': f"{r['ground_truth']:.2f}" if r['ground_truth'] else "N/A",
                'Predicted (m)': "Skipped" if r.get('skipped') else f"{r['prediction']:.2f}",
                'Error (m)': f"±{r['error']:.2f}" if r.get('error') else "N/A",
                'Error (%)': f"{r['error_percent']:.1f}" if r.get('error_percent') else "N/A"
            })
        
        st.dataframe(table_data, use_container_width=True, height=300)
        
        # EXPORT
        st.markdown("---")
        
        if st.button("Export Results (JSON)", use_container_width=False):
            json_str = json.dumps(st.session_state.results, indent=2, default=str)
            st.download_button(
                label="Download JSON File",
                data=json_str,
                file_name="distance_estimation_results.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
