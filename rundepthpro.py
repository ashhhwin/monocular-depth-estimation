"""
runDepthPro.py
Extract depth features from images using DepthPro model
"""

import os
import json
import numpy as np
import torch
import gc
from PIL import Image
from tqdm import tqdm
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation


class DepthProExtractor:
    """Extract depth features using DepthPro with caching"""
    
    def __init__(self, device='cuda', cache_dir=None, use_cache=True):
        """
        Initialize DepthPro extractor
        
        Args:
            device: 'cuda' or 'cpu'
            cache_dir: Directory to cache depth features (optional)
            use_cache: Whether to use caching
        """
        self.device = device
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load model
        print("Loading DepthPro model...")
        self.processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf")
        self.model = self.model.to(self.device).half().eval()
        print("✅ DepthPro ready\n")
    
    def extract_depth_features(self, img_path, bbox):
        """
        Extract 6 depth features from image at bbox location
        
        Args:
            img_path: Path to image
            bbox: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            np.array: [bottom_center_depth, mid_bottom_depth, mean_depth, 
                      median_depth, center_depth, std_depth]
        """
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Run DepthPro
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            for k in inputs:
                inputs[k] = inputs[k].half()
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            depth_map = self.processor.post_process_depth_estimation(
                outputs, target_sizes=[(img.height, img.width)]
            )[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
            
            # Extract bbox region
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(depth_map.shape[1], x2)
            y2 = min(depth_map.shape[0], y2)
            
            bbox_depth = depth_map[y1:y2, x1:x2]
            
            if bbox_depth.size == 0:
                return None
            
            # Calculate key points
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            bottom_y = y2 - 1  # Bottom of bbox
            mid_bottom_y = (center_y + bottom_y) // 2  # Midpoint between center and bottom
            
            # Clamp coordinates to valid range
            center_x = min(center_x, depth_map.shape[1] - 1)
            center_y = min(center_y, depth_map.shape[0] - 1)
            bottom_y = min(bottom_y, depth_map.shape[0] - 1)
            mid_bottom_y = min(mid_bottom_y, depth_map.shape[0] - 1)
            
            # Extract 6 depth features
            bottom_center_depth = float(depth_map[bottom_y, center_x])
            mid_bottom_depth = float(depth_map[mid_bottom_y, center_x])
            mean_depth = float(np.mean(bbox_depth))
            median_depth = float(np.median(bbox_depth))
            center_depth = float(depth_map[center_y, center_x])
            std_depth = float(np.std(bbox_depth))
            
            depth_features = np.array([
                bottom_center_depth,
                mid_bottom_depth,
                mean_depth,
                median_depth,
                center_depth,
                std_depth
            ], dtype=np.float32)
            
            # Cleanup
            del img, inputs, outputs, depth_map
            torch.cuda.empty_cache()
            gc.collect()
            
            return depth_features
            
        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {e}")
            return None
    
    def process_from_detection_json(self, detection_json_path, image_folder, output_json_path=None):
        """
        Process images from detection JSON
        
        Args:
            detection_json_path: Path to detection JSON (e.g., train_detections.json)
            image_folder: Path to folder containing images
            output_json_path: Where to save depth features JSON (optional)
            
        Returns:
            dict: {image_name: depth_features}
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING DEPTH FEATURES")
        print(f"{'='*70}")
        print(f"Detection JSON: {detection_json_path}")
        print(f"Image folder: {image_folder}")
        
        # Load detection JSON
        with open(detection_json_path, 'r') as f:
            detection_data = json.load(f)
        
        print(f"✅ Loaded {len(detection_data)} detections")
        
        # Process each image
        depth_results = {}
        skipped = 0
        
        for img_name, detection in tqdm(detection_data.items(), desc="Extracting depth"):
            # Check if image has lead vehicle
            if not detection.get('lead_vehicle') or not detection['lead_vehicle'].get('in_roi'):
                skipped += 1
                continue
            
            # Get bbox
            bbox = detection['lead_vehicle']['bbox']
            img_path = os.path.join(image_folder, img_name)
            
            # Check cache
            cache_key = img_name.replace('.jpg', '.npy').replace('.png', '.npy')
            cache_path = os.path.join(self.cache_dir, cache_key) if self.use_cache and self.cache_dir else None
            
            depth_features = None
            
            # Try loading from cache
            if cache_path and os.path.exists(cache_path):
                try:
                    depth_features = np.load(cache_path)
                except:
                    pass
            
            # Extract if not cached
            if depth_features is None:
                depth_features = self.extract_depth_features(img_path, bbox)
                
                # Cache it
                if depth_features is not None and cache_path:
                    np.save(cache_path, depth_features)
            
            if depth_features is not None:
                depth_results[img_name] = {
                    'depth_features': depth_features.tolist(),
                    'ground_truth_distance': detection.get('ground_truth_distance'),
                    'bbox': bbox
                }
        
        print(f"\n✅ Extracted depth features for {len(depth_results)} images")
        print(f"⚠️  Skipped {skipped} images (no lead vehicle)")
        
        # Save results
        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(depth_results, f, indent=2)
            print(f"✅ Saved to: {output_json_path}")
        
        return depth_results
    
    def process_split(self, split_name, detection_json_dir, image_base_dir, output_dir=None):
        """
        Process a data split (train/val/test)
        
        Args:
            split_name: 'train', 'val', or 'test'
            detection_json_dir: Directory containing detection JSONs
            image_base_dir: Base directory containing Train/Val/Test folders
            output_dir: Directory to save depth JSONs (optional)
            
        Returns:
            dict: Depth results
        """
        # Construct paths
        detection_json_path = os.path.join(detection_json_dir, f"{split_name}_detections.json")
        image_folder = os.path.join(image_base_dir, split_name.capitalize())
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_json_path = os.path.join(output_dir, f"{split_name}_depth_features.json")
        else:
            output_json_path = None
        
        # Process
        return self.process_from_detection_json(
            detection_json_path=detection_json_path,
            image_folder=image_folder,
            output_json_path=output_json_path
        )
    
    def cleanup(self):
        """Cleanup model from memory"""
        del self.model, self.processor
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ Cleaned up DepthPro model")


def extract_depth_features_for_splits(
    splits=['train', 'val', 'test'],
    detection_json_dir='lead_vehicle_features',
    image_base_dir='2025_UChicago_Distance_Data/final_preprocessed',
    output_dir='depth_features',
    cache_dir='depth_cache',
    device='cuda'
):
    """
    Convenience function to extract depth features for all splits
    
    Args:
        splits: List of splits to process
        detection_json_dir: Directory with detection JSONs
        image_base_dir: Base directory with Train/Val/Test folders
        output_dir: Where to save depth JSONs
        cache_dir: Where to cache depth features
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: {split_name: depth_results}
    """
    extractor = DepthProExtractor(device=device, cache_dir=cache_dir, use_cache=True)
    
    all_results = {}
    
    try:
        for split_name in splits:
            print(f"\n{'='*70}")
            print(f"PROCESSING SPLIT: {split_name.upper()}")
            print(f"{'='*70}")
            
            results = extractor.process_split(
                split_name=split_name,
                detection_json_dir=detection_json_dir,
                image_base_dir=image_base_dir,
                output_dir=output_dir
            )
            
            all_results[split_name] = results
    
    finally:
        extractor.cleanup()
    
    print(f"\n{'='*70}")
    print("ALL SPLITS COMPLETE")
    print(f"{'='*70}")
    for split_name, results in all_results.items():
        print(f"{split_name}: {len(results)} images processed")
    
    return all_results


# --- NEW FUNCTION FOR EXTERNAL CALLS ---

def extract_depth_features_custom(
    detection_json_path,
    image_folder,
    output_json_path=None,
    cache_dir='custom_depth_cache',
    device='cuda'
):
    """
    Extract depth features for a custom set of images defined by a detection JSON.
    
    Args:
        detection_json_path: Path to the detection JSON file.
        image_folder: Path to the directory containing the images.
        output_json_path: Optional path to save the final depth features JSON.
        cache_dir: Directory for storing intermediate depth features.
        device: 'cuda' or 'cpu'.
        
    Returns:
        dict: {image_name: depth_features}
    """
    
    # 1. Initialize the DepthProExtractor
    extractor = DepthProExtractor(device=device, cache_dir=cache_dir, use_cache=True)
    
    all_results = {}
    
    try:
        print(f"\n{'='*70}")
        print(f"PROCESSING CUSTOM DATASET")
        print(f"{'='*70}")
        
        # 2. Call the core processing method
        all_results = extractor.process_from_detection_json(
            detection_json_path=detection_json_path,
            image_folder=image_folder,
            output_json_path=output_json_path # Passes the path to save the output
        )
        
    finally:
        # 3. Ensure cleanup is always performed
        extractor.cleanup()
        
    print(f"\n✅ Custom extraction complete. Processed {len(all_results)} images.")
    return all_results

# Example usage
if __name__ == "__main__":
    # Extract depth features for all splits
    results = extract_depth_features_for_splits(
        splits=['train', 'val', 'test'],
        detection_json_dir='~/anu_try/HOPEFULLY_FINAL/gcs-bucket/lead_vehicle_features',
        image_base_dir='~/anu_try/HOPEFULLY_FINAL/gcs-bucket/2025_UChicago_Distance_Data/final_preprocessed',
        output_dir='~/anu_try/HOPEFULLY_FINAL/gcs-bucket/depth_features',
        cache_dir='~/anu_try/HOPEFULLY_FINAL/depth_cache',
        device='cuda'
    )



# -- SAMPLE USAGE OF CUSTOM FUNCTION --
'''

import os
# Assuming your library file is named 'runDepthPro.py'
from runDepthPro import extract_depth_features_custom 

# --- Define Your Custom Inputs ---

# 1. Path to the JSON containing image names and 'lead_vehicle' detections/bboxes.
DETECTION_JSON_PATH = '/path/to/my/new_data/custom_detections.json'

# 2. Path to the folder containing the actual image files referenced in the JSON.
IMAGE_DIR = '/path/to/my/new_data/images_folder'

# 3. Path where the final JSON output of depth features should be saved.
OUTPUT_JSON_PATH = '/path/to/my/output/custom_features_output.json'

# 4. Optional: Directory to use for caching (features are saved here as .npy files).
CACHE_DIR = '/tmp/my_temp_cache/depth_cache'

# 5. Device to run on.
DEVICE = 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu'

# --- Call the Function ---
print(f"Starting depth feature extraction on custom data...")

# Call the imported function, passing all your specific inputs
custom_results = extract_depth_features_custom(
    detection_json_path=DETECTION_JSON_PATH,
    image_folder=IMAGE_DIR,
    output_json_path=OUTPUT_JSON_PATH,
    cache_dir=CACHE_DIR,
    device=DEVICE
)

print("\n--- Results Summary ---")
print(f"Total features extracted: {len(custom_results)}")
print(f"Output saved to: {OUTPUT_JSON_PATH}")


'''

