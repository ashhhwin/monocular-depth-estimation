"""
Configuration File for Image Preprocessing

This file contains all the parameters used by the preprocessing.py script.
You can adjust these "dials" to change the behavior of the processing
without modifying the core logic.
"""

import numpy as np

# --- 1. Camera Undistortion Parameters ---
CameraMat = np.array([[2429.865965, 0.0, 1209.084876],
                      [0.0, 2424.492001, 1032.478074],
                      [0.0, 0.0, 1.0]])

DistCoeff = np.array([-0.393931, 0.185580, 0.000120, 0.000002, 0.0])

# Alpha parameter for undistortion.
# 1.0 = Keep all pixels, results in black borders (padding).
# 0.0 = Crop to valid pixels only, no black borders.
UNDISTORT_ALPHA = 0.0

# --- 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- 3. Brightness & Contrast ---
# Formula: new_image = (CONTRAST_ALPHA * old_image) + BRIGHTNESS_BETA
# 1.0 = No change in contrast
# 0 = No change in brightness
CONTRAST_ALPHA = 1.1   # Slightly increase contrast
BRIGHTNESS_BETA = 5      # Slightly increase brightness
