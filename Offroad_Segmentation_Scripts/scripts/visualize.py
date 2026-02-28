"""
Visualization script for DeepLabV3+ ResNet50 segmentation model
Generates side-by-side comparisons:
[Original | Ground Truth | Prediction | Overlay]
"""

import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import cv2
from tqdm import tqdm
import multiprocessing

# ==========================
# Configuration
# ==========================

n_classes = 10
batch_size = 1
w = 512
h = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
experiment_root = os.path.abspath(os.path.join(script_dir, ".."))

model_path = os.path.join(experiment_root, "models", "segmentation_head.pth")

val_dir = os.path.join(project_root, "Offroad_Segmentation_Training_Dataset", "val")
image_dir = os.path.join(val_dir, "Color_Images")
mask_dir = os.path.join(val_dir, "Segmentation")

results_dir = os.path.join(experiment_root, "result")
vis_dir = os.path.join(results_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)

# ==========================
# Class Mapping (MUST match training)
# ==========================

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

# ==========================
# Color Map (must match test script)
# ==========================

COLOR_MAP = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 128, 0),
    3: (128, 128, 0),
    4: (0, 0, 128),
    5: (128, 0, 128),
    6: (0, 128, 128),
    7: (128, 128, 128),
    8: (64, 0, 0),
    9: (192, 0, 0),
}

def decode_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        color_mask[mask == cls] = color
    return color_mask

# ==========================
# Dataset
# ==========================

class VisualDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image_np = np.array(image).astype(np.uint8)
        mask_np = convert_mask(mask)

        if self.transform:
            image = self.transform(image)

        return image, image_np, mask_np, name

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# Main
# ==========================

def main():

    dataset = VisualDataset(image_dir, mask_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    print(f"Validation samples: {len(dataset)}")

    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=n_classes,
    ).to(device)

    model = model.to(memory_format=torch.channels_last)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")

    with torch.no_grad():
        for img_tensor, img_np, mask_np, name in tqdm(loader):

            img_tensor = img_tensor.to(device)
            img_tensor = img_tensor.to(memory_format=torch.channels_last)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output = model(img_tensor)

            pred = torch.argmax(output, dim=1).cpu().numpy()

            for i in range(img_tensor.size(0)):

                original_img = img_np[i].numpy().astype(np.uint8)
                original_img = cv2.resize(original_img, (w, h))

                gt_mask = mask_np[i].numpy()
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                gt_color = decode_mask(gt_mask)
                pred_color = decode_mask(pred[i])

                overlay = cv2.addWeighted(original_img, 0.6, pred_color, 0.4, 0)

                combined = np.concatenate(
                    [original_img, gt_color, pred_color, overlay],
                    axis=1
                )

                save_path = os.path.join(vis_dir, name[i])
                cv2.imwrite(save_path, combined)
        print(f"\nVisualizations saved to: {vis_dir}")
        print("Done!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()