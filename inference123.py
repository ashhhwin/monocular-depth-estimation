# inference123.py
import os
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# ---------------- CONFIG ----------------
IMAGES_DIR = "input_images"
MODEL_WEIGHTS = "depth_calib_output/model_weights_final.pth"
USE_VIT = True
BACKBONE_NAME = "vit_base_patch16_224" if USE_VIT else "resnet50"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

# ----------------- Load DepthPro ----------------
depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
depth_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(DEVICE)

def get_depth_map(pil_img: Image.Image) -> np.ndarray:
    inputs = depth_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = depth_model(**inputs)
    post_processed = depth_processor.post_process_depth_estimation(
        outputs, target_sizes=[(pil_img.height, pil_img.width)],
    )
    dm = post_processed[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
    return dm

# ----------------- Load Backbone ----------------
print(f"Loading backbone {BACKBONE_NAME}...")
backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0, global_pool="avg")
backbone.eval().to(DEVICE)
for param in backbone.parameters():
    param.requires_grad = False
feat_dim = backbone.num_features
print("Backbone feature dim:", feat_dim)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

# ----------------- MLP model ----------------
class CombinedRegressor(nn.Module):
    def __init__(self, feat_dim, meta_dim=8, hidden=512):
        super().__init__()
        self.img_proj = nn.Linear(feat_dim, hidden)
        self.meta_proj = nn.Linear(meta_dim, hidden//4)
        self.head = nn.Sequential(
            nn.Linear(hidden + (hidden//4), hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, feat, meta):
        f = torch.relu(self.img_proj(feat))
        m = torch.relu(self.meta_proj(meta))
        x = torch.cat([f, m], dim=1)
        out = self.head(x)
        return out

model = CombinedRegressor(feat_dim=feat_dim, meta_dim=8, hidden=512).to(DEVICE)
state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("Loaded trained depth correction model.")

# ----------------- Random image selection ----------------
all_images = list(Path(IMAGES_DIR).glob("*.*"))
if not all_images:
    raise RuntimeError("No images found!")
img_path = random.choice(all_images)
print("Selected image:", img_path)

img_cv2 = cv2.imread(str(img_path))
img_disp = img_cv2.copy()
h, w = img_cv2.shape[:2]

# ----------------- Interactive bbox ----------------
bbox = {"x1": None, "y1": None, "x2": None, "y2": None, "drawing": False}
window_name = "Draw bbox - drag, release. 's' save, 'r' reset, 'q' quit"

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox["drawing"] = True
        bbox["x1"], bbox["y1"] = x, y
        bbox["x2"], bbox["y2"] = x, y
    elif event == cv2.EVENT_MOUSEMOVE and bbox["drawing"]:
        bbox["x2"], bbox["y2"] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        bbox["drawing"] = False
        bbox["x2"], bbox["y2"] = x, y

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, mouse_callback)

print("Draw a bbox around the object of interest. Press 's' to confirm, 'r' to reset, 'q' to quit.")

while True:
    disp = img_disp.copy()
    if bbox["x1"] is not None and bbox["x2"] is not None:
        cv2.rectangle(disp, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0,255,0), 2)
    cv2.imshow(window_name, disp)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('s'):
        print("BBox saved:", bbox)
        break
    elif key == ord('r'):
        bbox = {"x1": None, "y1": None, "x2": None, "y2": None, "drawing": False}
        print("Reset bbox.")
    elif key == ord('q'):
        print("Quitting.")
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# ----------------- Extract depth and features ----------------
x1, y1 = max(0, min(bbox["x1"], bbox["x2"])), max(0, min(bbox["y1"], bbox["y2"]))
x2, y2 = min(w-1, max(bbox["x1"], bbox["x2"])), min(h-1, max(bbox["y1"], bbox["y2"]))

pil_img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
depth_map = get_depth_map(pil_img)
bbox_crop = depth_map[y1:y2+1, x1:x2+1]

depth_center = float(bbox_crop[bbox_crop.shape[0]//2, bbox_crop.shape[1]//2])
depth_min = float(np.nanmin(bbox_crop))
depth_mean = float(np.nanmean(bbox_crop))
print(f"Depth stats - center: {depth_center:.2f}, min: {depth_min:.2f}, mean: {depth_mean:.2f}")

# Image features
img_crop = img_cv2[y1:y2+1, x1:x2+1][:,:,::-1]  # BGR->RGB
img_tensor = transform(img_crop).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    img_feat = backbone(img_tensor).squeeze(0)

# BBox normalized features
bbox_w = (x2 - x1 + 1)/w
bbox_h = (y2 - y1 + 1)/h
bbox_cx = (x1 + x2)/2.0 / w
bbox_cy = (y1 + y2)/2.0 / h
bbox_by = y2 / h
meta_feat = torch.tensor([depth_center, depth_mean, depth_min, bbox_w, bbox_h, bbox_cx, bbox_cy, bbox_by], dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ----------------- Predict corrected depth ----------------
with torch.no_grad():
    pred_depth = model(img_feat.unsqueeze(0), meta_feat).item()

print(f"\nFinal corrected depth prediction: {pred_depth:.2f} meters")