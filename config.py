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


# --- 3. Gamma Correction (Non-linear Brightness) ---

# > 1.0 : Brightens shadows/mid-tones (good for day/shadows)
# < 1.0 : Darkens mid-tones (good for over-exposed/fog)
# 1.0 : No change
#
# A good value for shadow detail is 1.5
GAMMA = 1.5
