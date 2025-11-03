"""
Image Preprocessing Pipeline

This script provides functions to clean and enhance dashcam images.
It is designed to be both:
  1. A standalone script to batch-process folders.
  2. A module to be imported into other scripts (e.g., training, inference).

Functions:
- undistort_image(image): Removes camera lens distortion.
- apply_white_balance(image): Corrects color cast using Gray World algorithm.
- apply_clahe(image): Enhances local contrast.
- apply_gamma_correction(image): Adjusts non-linear brightness (good for shadows).
- preprocess_image(image): Runs the full pipeline on a single image.

Standalone Usage:
    python preprocessing.py /path/to/input_folder /path/to/output_folder
"""

import cv2
import numpy as np
import os
import argparse
from glob import glob
from tqdm import tqdm

# Import all parameters from our config file
try:
    import config
except ImportError:
    print("ERROR: config.py not found!")
    print("Please make sure config.py is in the same directory.")
    exit(1)


def undistort_image(image: np.ndarray) -> np.ndarray:
    """
    Removes lens distortion from an image using camera parameters
    from the config file.
    
    Args:
        image: The input image (NumPy array).
        
    Returns:
        The undistorted (and potentially cropped) image.
    """
    h, w = image.shape[:2]
    
    # Get the new optimal camera matrix based on the ALPHA
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        config.CameraMat, 
        config.DistCoeff, 
        (w, h), 
        config.UNDISTORT_ALPHA, 
        (w, h)
    )
    
    # Undistort the image
    undistorted = cv2.undistort(
        image, 
        config.CameraMat, 
        config.DistCoeff, 
        None, 
        new_camera_matrix
    )
    
    # Crop the image based on the ROI if alpha=0
    # Note: With alpha=0, roi contains the valid pixel area
    if config.UNDISTORT_ALPHA == 0:
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
    return undistorted


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Applies the "Gray World" white balance algorithm.
    Assumes the average color of the entire scene is gray.
    
    Args:
        image: The input image (NumPy array).
        
    Returns:
        The white-balanced image.
    """
    # Split the image into its B, G, R channels
    b, g, r = cv2.split(image)
    
    # Calculate the average of each channel
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    # Calculate the overall average
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    # Calculate the scaling factor for each channel
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    
    # Scale the channels
    b_balanced = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g_balanced = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r_balanced = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    # Merge the balanced channels back together
    return cv2.merge([b_balanced, g_balanced, r_balanced])


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance local contrast.
    
    Args:
        image: The input image (NumPy array).
        
    Returns:
        The contrast-enhanced image.
    """
    # Convert to LAB color space to apply CLAHE only to the Lightness channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create the CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    
    # Apply CLAHE to the L-channel
    l_clahe = clahe.apply(l)
    
    # Merge the channels back and convert to BGR
    lab_enhanced = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image: np.ndarray) -> np.ndarray:
    """
    Applies non-linear gamma correction to the image.
    
    This is superior to linear brightness/contrast as it brightens
    shadows and mid-tones more than highlights, preventing blowouts.
    
    Args:
        image: The input image (NumPy array).
        
    Returns:
        The gamma-corrected image.
    """
    # Build a lookup table (LUT) mapping pixel values [0, 255]
    # to their new gamma-corrected values.
    # We use config.GAMMA, but apply the inverse formula.
    inv_gamma = 1.0 / config.GAMMA
    
    # This creates a 256-element array
    lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    
    # Apply the LUT to every pixel in the image (very fast)
    return cv2.LUT(image, lut)


# --- The Main Pipeline Function ---

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Runs the full preprocessing pipeline on a single image.
    This is the function you'll likely import into your training/inference scripts.
    
    Args:
        image: The raw input image (NumPy array).
        
    Returns:
        The fully preprocessed image (NumPy array).
    """
    # Step 1: Fix geometric distortion
    processed = undistort_image(image)
    
    # Step 2: Fix color cast
    processed = apply_white_balance(processed)
    
    # Step 3: Enhance local contrast
    processed = apply_clahe(processed)
    
    # Step 4: Final non-linear brightness adjustment
    processed = apply_gamma_correction(processed)
    
    return processed


# --- Standalone Script Execution ---

if __name__ == "__main__":
    """
    This block runs when you execute the script directly from the terminal.
    It processes an entire folder of images.
    """
    parser = argparse.ArgumentParser(
        description="Batch preprocess images from a folder."
    )
    parser.add_argument(
        "input_dir", 
        type=str, 
        help="Path to the folder containing raw images."
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        help="Path to the folder where processed images will be saved."
    )
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        exit(1)
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting batch preprocessing...")
    print(f"Input folder:  {args.input_dir}")
    print(f"Output folder: {args.output_dir}")
    
    # Find all images
    image_paths = glob(os.path.join(args.input_dir, '*.jpg')) + \
                  glob(os.path.join(args.input_dir, '*.jpeg')) + \
                  glob(os.path.join(args.input_dir, '*.png'))
                  
    if not image_paths:
        print("Error: No .jpg, .jpeg, or .png images found.")
        exit(1)
        
    print(f"Found {len(image_paths)} images to process.")
    
    # Process each image with a progress bar
    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # 1. Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"\nWarning: Failed to read {img_path}. Skipping.")
                continue
                
            # 2. Run the pipeline
            processed_image = preprocess_image(image)
            
            # 3. Save the result
            filename = os.path.basename(img_path)
            output_path = os..path.join(args.output_dir, filename)
            cv2.imwrite(output_path, processed_image)
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}. Skipping.")
            
    print("\nBatch processing complete.")
    print(f"Processed images saved to: {args.output_dir}")
