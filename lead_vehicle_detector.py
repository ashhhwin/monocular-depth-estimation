"""
Lead Vehicle Detection for Dashcam Images

This module provides functionality to detect and track the lead vehicle in dashcam footage.
It uses YOLOv8 for vehicle detection and adaptive ROI based on lane line detection.

Usage as standalone script:
    python lead_vehicle_detector.py

Usage as importable module:
    from lead_vehicle_detector import process_dashcam_images
    
    # Basic usage - just JSON output
    results = process_dashcam_images(
        input_dir="dashcam_images/",
        output_json="results.json"
    )
    
    # Full usage - with visualizations and CSV
    results = process_dashcam_images(
        input_dir="dashcam_images/",
        output_json="results.json",
        save_visualizations=True,
        output_viz_dir="output_annotated/",
        save_csv=True,
        output_csv="results.csv",
        conf_threshold=0.30,
        use_adaptive_roi=True
    )
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import csv
from typing import Dict, List, Optional, Tuple


class LeadVehicleDetector:
    def __init__(self, model_path='yolov8l.pt', hood_exclude_ratio=0.09, 
                 conf_threshold=0.20, preprocess_image=False, use_adaptive_roi=True,
                 device=None):
        """
        Initialize lead vehicle detector
        
        Args:
            model_path: Path to YOLO model (default: yolov8l.pt)
            hood_exclude_ratio: Bottom portion of image to exclude (0.09 = 9%)
            conf_threshold: YOLO confidence threshold (default: 0.20)
            preprocess_image: Whether to enhance image contrast before detection
            use_adaptive_roi: Whether to adjust ROI based on lane/vanishing point detection
            device: Device to run inference on. None = auto-detect (cuda > mps > cpu)
        """
        # Auto-detect best available device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print("Using CUDA (NVIDIA GPU)")
            elif torch.backends.mps.is_available():
                device = 'mps'
                print("Using MPS (Apple Silicon)")
            else:
                device = 'cpu'
                print("Using CPU")
        else:
            print(f"Using specified device: {device}")
        
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        
        self.hood_exclude_ratio = hood_exclude_ratio
        self.conf_threshold = conf_threshold
        self.preprocess_image = preprocess_image
        self.use_adaptive_roi = use_adaptive_roi
        
        # Base ROI polygon in absolute pixel coordinates
        self.base_roi_polygon = np.array([
            [329, 2047],      # Bottom-left
            [2152, 2047],     # Bottom-right
            [1287, 921],      # Top-right
            [1159, 921]       # Top-left
        ], dtype=np.int32)
        
        # Base vanishing point (center of top edge of ROI)
        self.base_vanishing_x = (1287 + 1159) // 2
        
        # Smoothing for shift values
        self.shift_history = []
        self.max_history = 5
        
        # Tracking state
        self.prev_lead_bbox = None
        self.frame_count = 0
        
    def enhance_image(self, img):
        """
        Enhance image contrast and brightness for better detection
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception as e:
            print(f"Warning: Image enhancement failed: {e}")
            return img
    
    def detect_lane_lines(self, img):
        """Detect lane lines using edge detection and Hough transform"""
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            
            mask = np.zeros_like(edges)
            roi_vertices = np.array([[
                (int(w * 0.1), h),
                (int(w * 0.9), h),
                (int(w * 0.6), int(h * 0.6)),
                (int(w * 0.4), int(h * 0.6))
            ]], dtype=np.int32)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            lines = cv2.HoughLinesP(
                masked_edges, rho=2, theta=np.pi/180,
                threshold=50, minLineLength=40, maxLineGap=100
            )
            
            if lines is None or len(lines) == 0:
                return self.base_vanishing_x, []
            
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if 0.3 < abs(slope) < 3:
                    if slope < 0:
                        left_lines.append(line[0])
                    else:
                        right_lines.append(line[0])
            
            vanishing_x = self.base_vanishing_x
            
            if len(left_lines) > 0 and len(right_lines) > 0:
                left_fit = self.fit_lane_line(left_lines, h)
                right_fit = self.fit_lane_line(right_lines, h)
                
                if left_fit is not None and right_fit is not None:
                    x_intersect = self.find_line_intersection(left_fit, right_fit)
                    if x_intersect is not None:
                        vanishing_x = int(x_intersect)
            
            return vanishing_x, lines
            
        except Exception as e:
            print(f"Warning: Lane detection failed: {e}")
            return self.base_vanishing_x, []
    
    def fit_lane_line(self, lines, img_height):
        """Fit a line through detected lane segments"""
        if len(lines) == 0:
            return None
        points = []
        for x1, y1, x2, y2 in lines:
            points.append([x1, y1])
            points.append([x2, y2])
        points = np.array(points)
        try:
            x = points[:, 0]
            y = points[:, 1]
            coeffs = np.polyfit(y, x, 1)
            return coeffs
        except:
            return None
    
    def find_line_intersection(self, line1_coeffs, line2_coeffs):
        """Find x-coordinate where two lines intersect at top of ROI"""
        try:
            m1, b1 = line1_coeffs
            m2, b2 = line2_coeffs
            y_intersect = 921
            x1 = m1 * y_intersect + b1
            x2 = m2 * y_intersect + b2
            vanishing_x = (x1 + x2) / 2
            return vanishing_x
        except:
            return None
    
    def calculate_roi_shift(self, vanishing_x):
        """Calculate horizontal shift based on vanishing point"""
        deviation = vanishing_x - self.base_vanishing_x
        shift = int(deviation * 0.6)
        shift = np.clip(shift, -100, 100)
        
        self.shift_history.append(shift)
        if len(self.shift_history) > self.max_history:
            self.shift_history.pop(0)
        
        if len(self.shift_history) > 0:
            weights = np.exp(np.linspace(-1, 0, len(self.shift_history)))
            weights /= weights.sum()
            smoothed_shift = int(np.average(self.shift_history, weights=weights))
            return smoothed_shift
        
        return shift
    
    def get_adaptive_roi(self, img, horizontal_shift=0):
        """Get ROI polygon, adjusted for turns based on vanishing point"""
        roi = self.base_roi_polygon.copy()
        if horizontal_shift != 0:
            roi[:, 0] = roi[:, 0] + horizontal_shift
            h, w = img.shape[:2]
            roi[:, 0] = np.clip(roi[:, 0], 0, w)
        return roi
    
    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        result = cv2.pointPolygonTest(polygon, tuple(point), False)
        return result >= 0
    
    def filter_hood_detections(self, detections, img_height):
        """Remove detections in bottom portion (hood area)"""
        hood_threshold = img_height * (1 - self.hood_exclude_ratio)
        filtered = []
        for det in detections:
            bbox = det['bbox']
            if bbox[3] < hood_threshold:
                filtered.append(det)
        return filtered
    
    def get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return np.array([cx, cy], dtype=np.float32)
    
    def calculate_closeness_score(self, bbox, img_height):
        """Calculate how close the vehicle is based on vertical position"""
        cy = (bbox[1] + bbox[3]) / 2
        norm_cy = cy / img_height
        if norm_cy > 0.8:
            return 1.0
        elif norm_cy > 0.5:
            return 0.5 + (norm_cy - 0.5) / 0.3 * 0.5
        else:
            return norm_cy
    
    def calculate_size_score(self, bbox, img_width, img_height):
        """Calculate size score - larger vehicles are typically closer"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        frame_area = img_width * img_height
        relative_size = area / frame_area
        if relative_size > 0.4:
            return 0.2
        elif relative_size > 0.03:
            return min(relative_size / 0.15, 1.0)
        else:
            return relative_size / 0.03 * 0.5
    
    def calculate_center_alignment_score(self, bbox, img_width):
        """Calculate how aligned the vehicle is with image center"""
        cx = (bbox[0] + bbox[2]) / 2
        center_deviation = abs(cx - img_width / 2) / (img_width / 2)
        if center_deviation < 0.15:
            return 1.0
        elif center_deviation < 0.3:
            return 1.0 - (center_deviation - 0.15) / 0.15 * 0.5
        else:
            return 0.5 - min((center_deviation - 0.3) / 0.3, 0.4)
    
    def find_lead_vehicle(self, image_path, visualize=False):
        """
        Find the lead vehicle in a single image
        
        Returns:
            dict: {
                'filename': str,
                'ground_truth_distance': float or None (extracted from filename),
                'all_vehicles': list of dicts,
                'lead_vehicle': dict or None,
                'metadata': dict with vanishing_point, roi_shift, etc.
            }
            annotated_img: if visualize=True
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"ERROR: Could not read image: {image_path}")
                return None, None
            
            h, w = img.shape[:2]
            filename = os.path.basename(image_path)
            
            # Extract ground truth distance from filename
            # Format: seq462_dist18.48_time1743693342.885320967.jpg
            ground_truth_distance = None
            try:
                if '_dist' in filename:
                    dist_part = filename.split('_dist')[1].split('_')[0]
                    ground_truth_distance = float(dist_part)
            except:
                pass  # If parsing fails, leave as None
            
            # Detect lanes and calculate shift
            horizontal_shift = 0
            vanishing_x = self.base_vanishing_x
            detected_lines = []
            
            if self.use_adaptive_roi:
                vanishing_x, detected_lines = self.detect_lane_lines(img)
                horizontal_shift = self.calculate_roi_shift(vanishing_x)
            
            roi_polygon = self.get_adaptive_roi(img, horizontal_shift)
            
            # Preprocess if enabled
            img_for_detection = self.enhance_image(img) if self.preprocess_image else img
            
            # Run YOLO detection
            results = self.model(img_for_detection, verbose=False, 
                               conf=self.conf_threshold, device=self.device)[0]
            
            # Filter for vehicle classes
            vehicle_classes = [2, 5, 7, 3]
            detections = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls in vehicle_classes:
                        bbox = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append({
                            'bbox': bbox,
                            'conf': conf,
                            'class': cls
                        })
            
            # Filter out hood detections
            detections = self.filter_hood_detections(detections, h)
            
            # Separate into ROI and non-ROI
            roi_detections = []
            non_roi_detections = []
            
            for det in detections:
                bbox = det['bbox']
                center = self.get_bbox_center(bbox)
                in_roi = self.point_in_polygon(center, roi_polygon)
                
                det['in_roi'] = in_roi
                det['bbox_list'] = bbox.tolist()  # Convert to list for JSON
                
                if in_roi:
                    roi_detections.append(det)
                else:
                    non_roi_detections.append(det)
            
            # Find lead vehicle
            lead_detection = None
            is_roi_vehicle = False
            
            if len(roi_detections) > 0:
                best_score = -1
                best_idx = -1
                
                for idx, det in enumerate(roi_detections):
                    bbox = det['bbox']
                    closeness = self.calculate_closeness_score(bbox, h)
                    size = self.calculate_size_score(bbox, w, h)
                    alignment = self.calculate_center_alignment_score(bbox, w)
                    score = closeness * 0.7 + size * 0.2 + alignment * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx >= 0:
                    lead_detection = roi_detections[best_idx]
                    is_roi_vehicle = True
            
            # Fallback
            if lead_detection is None and len(non_roi_detections) > 0:
                best_score = -1
                best_idx = -1
                
                for idx, det in enumerate(non_roi_detections):
                    bbox = det['bbox']
                    closeness = self.calculate_closeness_score(bbox, h)
                    size = self.calculate_size_score(bbox, w, h)
                    alignment = self.calculate_center_alignment_score(bbox, w)
                    score = closeness * 0.4 + size * 0.3 + alignment * 0.3
                    
                    if alignment > 0.3 and score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx >= 0:
                    lead_detection = non_roi_detections[best_idx]
                    is_roi_vehicle = False
            
            # Build result dict
            all_vehicles = []
            for det in detections:
                all_vehicles.append({
                    'bbox': det['bbox_list'],
                    'confidence': float(det['conf']),
                    'in_roi': det['in_roi']
                })
            
            lead_vehicle = None
            if lead_detection is not None:
                lead_vehicle = {
                    'bbox': lead_detection['bbox_list'],
                    'confidence': float(lead_detection['conf']),
                    'in_roi': lead_detection['in_roi'],
                    'is_lead': True
                }
                self.prev_lead_bbox = lead_detection['bbox']
            else:
                self.prev_lead_bbox = None
            
            metadata = {
                'vanishing_point_x': int(vanishing_x),
                'roi_shift': int(horizontal_shift),
                'num_lanes_detected': len(detected_lines) if detected_lines is not None and len(detected_lines) > 0 else 0,
                'total_vehicles': len(all_vehicles),
                'roi_vehicles': len(roi_detections),
                'image_dimensions': {'width': w, 'height': h}
            }
            
            result = {
                'filename': filename,
                'ground_truth_distance': ground_truth_distance,
                'all_vehicles': all_vehicles,
                'lead_vehicle': lead_vehicle,
                'metadata': metadata
            }
            
            self.frame_count += 1
            
            # Visualize if requested
            annotated_img = None
            if visualize:
                annotated_img = self._create_visualization(
                    img, roi_polygon, roi_detections, non_roi_detections,
                    lead_detection['bbox'] if lead_detection else None,
                    lead_detection['conf'] if lead_detection else 0,
                    is_roi_vehicle, horizontal_shift, vanishing_x, detected_lines, h, w
                )
            
            return result, annotated_img
            
        except Exception as e:
            print(f"ERROR processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _create_visualization(self, img, roi_polygon, roi_detections, non_roi_detections,
                             lead_bbox, lead_conf, is_roi_vehicle, horizontal_shift, 
                             vanishing_x, detected_lines, h, w):
        """Create detailed visualization with all detections"""
        annotated_img = img.copy()
        
        # Draw detected lane lines
        if detected_lines is not None and len(detected_lines) > 0:
            for line in detected_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw vanishing point
        cv2.circle(annotated_img, (vanishing_x, 921), 10, (0, 255, 0), -1)
        cv2.line(annotated_img, (vanishing_x, 0), (vanishing_x, h), (0, 255, 0), 1)
        cv2.putText(annotated_img, f"VP: {vanishing_x}", (vanishing_x + 15, 900), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw ROI
        color = (0, 255, 255)
        if horizontal_shift < -10:
            color = (0, 255, 0)
        elif horizontal_shift > 10:
            color = (0, 0, 255)
        
        cv2.polylines(annotated_img, [roi_polygon], True, color, 3)
        roi_label = f"ROI (shift: {horizontal_shift}px)" if horizontal_shift != 0 else "ROI"
        cv2.putText(annotated_img, roi_label, 
                   (roi_polygon[0][0] + 10, roi_polygon[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw hood exclusion
        hood_y = int(h * (1 - self.hood_exclude_ratio))
        cv2.line(annotated_img, (0, hood_y), (w, hood_y), (255, 0, 255), 2)
        
        # Draw ROI detections
        for det in roi_detections:
            bbox = det['bbox'].astype(int)
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), (255, 255, 0), 2)
            center = self.get_bbox_center(det['bbox']).astype(int)
            cv2.circle(annotated_img, tuple(center), 5, (255, 255, 0), -1)
        
        # Draw non-ROI detections
        for det in non_roi_detections:
            bbox = det['bbox'].astype(int)
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), (128, 128, 128), 2)
            center = self.get_bbox_center(det['bbox']).astype(int)
            cv2.circle(annotated_img, tuple(center), 5, (128, 128, 128), -1)
        
        # Draw lead vehicle
        if lead_bbox is not None:
            bbox = lead_bbox.astype(int)
            color = (0, 255, 0) if is_roi_vehicle else (0, 165, 255)
            thickness = 4 if is_roi_vehicle else 3
            
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), color, thickness)
            
            label = f"LEAD ({'ROI' if is_roi_vehicle else 'FALLBACK'}): {lead_conf:.2f}"
            cv2.putText(annotated_img, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            center = self.get_bbox_center(lead_bbox).astype(int)
            cv2.circle(annotated_img, tuple(center), 8, color, -1)
        
        # Draw center line
        center_x = w // 2
        cv2.line(annotated_img, (center_x, 0), (center_x, h), (255, 255, 255), 1)
        
        # Detection count
        cv2.putText(annotated_img, 
                   f"ROI: {len(roi_detections)} | Non-ROI: {len(non_roi_detections)}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_img


def process_dashcam_images(
    input_dir: str,
    output_json: str,
    save_visualizations: bool = False,
    output_viz_dir: Optional[str] = None,
    save_csv: bool = False,
    output_csv: Optional[str] = None,
    model_path: str = 'yolov8l.pt',
    hood_exclude_ratio: float = 0.09,
    conf_threshold: float = 0.25,
    preprocess_image: bool = False,
    use_adaptive_roi: bool = True,
    device: Optional[str] = None
) -> Dict:
    """
    Process dashcam images to detect lead vehicles and save results.
    
    Args:
        input_dir: Directory containing dashcam images
            Example: "dashcam_images/"
        
        output_json: Path to save JSON results
            Example: "results.json"
        
        save_visualizations: Whether to save annotated images with bounding boxes
            Example: True (default: False)
        
        output_viz_dir: Directory to save visualizations (required if save_visualizations=True)
            Example: "output_annotated/"
        
        save_csv: Whether to save results as CSV for easy analysis
            Example: True (default: False)
        
        output_csv: Path to save CSV (if None and save_csv=True, uses same name as JSON)
            Example: "results.csv"
        
        model_path: Path to YOLO model weights
            Example: 'yolov8l.pt' (large/accurate) or 'yolov8n.pt' (nano/faster)
        
        hood_exclude_ratio: Bottom portion of image to exclude (car hood)
            Example: 0.09 means exclude bottom 9% of image
        
        conf_threshold: YOLO confidence threshold for detections
            Example: 0.25 (lower = more detections but more false positives)
            Range: 0.0 to 1.0
        
        preprocess_image: Apply contrast enhancement before detection
            Example: False (set True for poor lighting/night conditions)
        
        use_adaptive_roi: Adjust ROI based on lane detection (handles curves)
            Example: True (recommended for varied road conditions)
        
        device: Device for inference ('cuda', 'mps', 'cpu', or None for auto)
            Example: None (auto-detect), 'mps' (Apple Silicon), 'cuda' (NVIDIA GPU)
    
    Returns:
        dict: Summary statistics including:
            - total_frames: int - Total number of images processed
            - frames_with_lead: int - Images where lead vehicle was detected
            - detection_rate: float - Percentage of successful detections
            - output_files: dict - Paths to all generated files
    
    JSON Output Format:
        {
            "frame_001.jpg": {
                "all_vehicles": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.85,
                        "in_roi": true
                    },
                    ...
                ],
                "lead_vehicle": {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.85,
                    "in_roi": true,
                    "is_lead": true
                } or null,
                "metadata": {
                    "vanishing_point_x": 1223,
                    "roi_shift": -15,
                    "num_lanes_detected": 2,
                    "total_vehicles": 3,
                    "roi_vehicles": 1,
                    "image_dimensions": {"width": 2464, "height": 2056}
                }
            },
            ...
        }
    
    Example Usage:
        # Basic usage - minimal parameters, just JSON output
        results = process_dashcam_images(
            input_dir="dashcam_images/",
            output_json="results.json"
        )
        
        # With visualizations - save annotated images
        results = process_dashcam_images(
            input_dir="dashcam_images/",
            output_json="results.json",
            save_visualizations=True,
            output_viz_dir="output_annotated/"
        )
        
        # Full options - visualizations, CSV, custom thresholds
        results = process_dashcam_images(
            input_dir="dashcam_images/",
            output_json="results.json",
            save_visualizations=True,
            output_viz_dir="output_annotated/",
            save_csv=True,
            output_csv="results.csv",
            conf_threshold=0.30,  # Higher threshold = fewer but more confident detections
            use_adaptive_roi=True,  # Adjust for road curves
            device='mps'  # Use Apple Silicon GPU
        )
        
        # Night/poor lighting conditions
        results = process_dashcam_images(
            input_dir="night_footage/",
            output_json="night_results.json",
            preprocess_image=True,  # Enable contrast enhancement
            conf_threshold=0.20  # Lower threshold for harder conditions
        )
        
        # Fast processing with smaller model
        results = process_dashcam_images(
            input_dir="dashcam_images/",
            output_json="results.json",
            model_path='yolov8n.pt',  # Nano model - much faster
            conf_threshold=0.30
        )
    """
    # Validate inputs
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if save_visualizations and output_viz_dir is None:
        raise ValueError("output_viz_dir must be specified when save_visualizations=True")
    
    # Create output directories
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    if save_visualizations:
        os.makedirs(output_viz_dir, exist_ok=True)
    
    # Initialize detector
    print(f"\n{'='*60}")
    print("Lead Vehicle Detector - Processing Started")
    print(f"{'='*60}")
    detector = LeadVehicleDetector(
        model_path=model_path,
        hood_exclude_ratio=hood_exclude_ratio,
        conf_threshold=conf_threshold,
        preprocess_image=preprocess_image,
        use_adaptive_roi=use_adaptive_roi,
        device=device
    )
    
    # Get all images
    image_paths = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    print(f"\nFound {len(image_paths)} images in '{input_dir}'")
    print(f"Configuration:")
    print(f"  - Model: {model_path}")
    print(f"  - Confidence Threshold: {conf_threshold}")
    print(f"  - Adaptive ROI: {use_adaptive_roi}")
    print(f"  - Preprocessing: {preprocess_image}")
    print(f"  - Save Visualizations: {save_visualizations}")
    print(f"  - Save CSV: {save_csv}")
    print(f"\nProcessing...")
    
    # Process all images
    all_results = {}
    csv_rows = []
    
    for i, img_path in enumerate(image_paths):
        result, annotated = detector.find_lead_vehicle(
            img_path, 
            visualize=save_visualizations
        )
        
        if result is not None:
            all_results[result['filename']] = {
                'ground_truth_distance': result['ground_truth_distance'],
                'all_vehicles': result['all_vehicles'],
                'lead_vehicle': result['lead_vehicle'],
                'metadata': result['metadata']
            }
            
            # Prepare CSV row
            if save_csv:
                lead = result['lead_vehicle']
                if lead is not None:
                    csv_rows.append({
                        'filename': result['filename'],
                        'ground_truth_distance': result['ground_truth_distance'] if result['ground_truth_distance'] is not None else '',
                        'lead_x1': lead['bbox'][0],
                        'lead_y1': lead['bbox'][1],
                        'lead_x2': lead['bbox'][2],
                        'lead_y2': lead['bbox'][3],
                        'lead_confidence': lead['confidence'],
                        'lead_in_roi': lead['in_roi'],
                        'total_vehicles': result['metadata']['total_vehicles']
                    })
                else:
                    csv_rows.append({
                        'filename': result['filename'],
                        'ground_truth_distance': result['ground_truth_distance'] if result['ground_truth_distance'] is not None else '',
                        'lead_x1': '',
                        'lead_y1': '',
                        'lead_x2': '',
                        'lead_y2': '',
                        'lead_confidence': '',
                        'lead_in_roi': '',
                        'total_vehicles': result['metadata']['total_vehicles']
                    })
            
            # Save visualization
            if save_visualizations and annotated is not None:
                out_path = os.path.join(output_viz_dir, result['filename'])
                cv2.imwrite(out_path, annotated)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(image_paths)} images ({(i+1)/len(image_paths)*100:.1f}%)")
    
    print(f"  Progress: {len(image_paths)}/{len(image_paths)} images (100.0%)")
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ JSON saved to: {output_json}")
    
    # Save CSV if requested
    csv_path = None
    if save_csv:
        if output_csv is None:
            csv_path = output_json.replace('.json', '.csv')
        else:
            csv_path = output_csv
        
        with open(csv_path, 'w', newline='') as f:
            if csv_rows:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
        print(f"✓ CSV saved to: {csv_path}")
    
    if save_visualizations:
        print(f"✓ Visualizations saved to: {output_viz_dir}")
    
    # Calculate summary statistics
    frames_with_lead = sum(1 for r in all_results.values() if r['lead_vehicle'] is not None)
    total_frames = len(all_results)
    detection_rate = (frames_with_lead / total_frames * 100) if total_frames > 0 else 0
    
    summary = {
        'total_frames': total_frames,
        'frames_with_lead': frames_with_lead,
        'detection_rate': detection_rate,
        'output_files': {
            'json': output_json,
            'csv': csv_path,
            'visualizations': output_viz_dir if save_visualizations else None
        }
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("Detection Summary")
    print(f"{'='*60}")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with lead vehicle: {frames_with_lead}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"{'='*60}\n")
    
    return summary


if __name__ == "__main__":
    """
    Standalone execution with example configuration.
    
    To customize, either:
    1. Edit the parameters below directly
    2. Or import this function in your own script:
       from lead_vehicle_detector import process_dashcam_images
    """
    
    print("\n" + "="*60)
    print("LEAD VEHICLE DETECTOR - Standalone Mode")
    print("="*60)
    print("\nRunning with default configuration...")
    print("To customize parameters, edit this file or import the function.\n")
    
    # Configuration - EDIT THESE VALUES
    input_dir = "batch_1/"              # Input directory with images
    output_json = "carla-batch-1-results.json"                # JSON output file
    save_visualizations = True                  # Save annotated images?
    output_viz_dir = "carla-batch-1-output_annotated/"        # Where to save annotated images
    save_csv = True                             # Also save as CSV?
    
    # Advanced options
    model_path = 'yolov8l.pt'                   # yolov8n.pt (fast) or yolov8l.pt (accurate)
    conf_threshold = 0.25                       # Detection confidence (0.0-1.0)
    use_adaptive_roi = True                     # Adjust for road curves
    preprocess_image = True                    # Enable for poor lighting
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\n⚠ ERROR: Input directory '{input_dir}' not found!")
        print(f"\nPlease either:")
        print(f"  1. Create the directory and add images")
        print(f"  2. Edit the 'input_dir' variable in this script")
        print(f"  3. Import and call process_dashcam_images() with your path\n")
        exit(1)
    
    # Run processing
    try:
        results = process_dashcam_images(
            input_dir=input_dir,
            output_json=output_json,
            save_visualizations=save_visualizations,
            output_viz_dir=output_viz_dir,
            save_csv=save_csv,
            model_path=model_path,
            conf_threshold=conf_threshold,
            preprocess_image=preprocess_image,
            use_adaptive_roi=use_adaptive_roi
        )
        
        print("✓ Processing completed successfully!")
        print(f"\nOutput files generated:")
        if results['output_files']['json']:
            print(f"  - JSON: {results['output_files']['json']}")
        if results['output_files']['csv']:
            print(f"  - CSV: {results['output_files']['csv']}")
        if results['output_files']['visualizations']:
            print(f"  - Visualizations: {results['output_files']['visualizations']}")
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)