"""
Inference script for DeepLabV3+ EfficientNet-B4 segmentation model
"""

import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm
import multiprocessing

# ==========================
# Configuration
# ==========================

n_classes = 10
batch_size = 6
w = 512
h = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # speed boost for fixed image size

print(f"Using device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to saved model
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
experiment_root = os.path.abspath(os.path.join(script_dir, ".."))
model_path = os.path.join(experiment_root,"models","segmentation_head.pth")

# Path to test images
test_dir = os.path.join(project_root,"Offroad_Segmentation_testImages")
image_dir = os.path.join(project_root, "Offroad_Segmentation_testImages", "Color_Images")

# Where predictions will be saved
results_dir = os.path.join(experiment_root, "result")
output_dir = os.path.join(results_dir, "test_prediction")
os.makedirs(output_dir, exist_ok=True)

# ==========================
# Dataset
# ==========================

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Example color map
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

COLOR_LUT = np.zeros((n_classes, 3), dtype=np.uint8)
for k, v in COLOR_MAP.items():
    COLOR_LUT[k] = v

def main():
    test_dataset = TestDataset(image_dir=image_dir, transform=transform)
    num_workers = 2

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    print(f"Test samples: {len(test_dataset)}")

    # ==========================
    # Load Model
    # ==========================

    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,   # we load trained weights
        in_channels=3,
        classes=n_classes,
    ).to(device)

    model = model.to(memory_format=torch.channels_last)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")

    
    # ==========================
    # Inference
    # ==========================

    with torch.no_grad():
        for imgs, img_names in tqdm(test_loader, desc="Running Inference"):

            imgs = imgs.to(device, non_blocking=True)
            imgs = imgs.to(memory_format=torch.channels_last)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):

                outputs = model(imgs)

                # Horizontal Flip TTA
                flipped_imgs = torch.flip(imgs, dims=[3])
                flipped_outputs = model(flipped_imgs)
                flipped_outputs = torch.flip(flipped_outputs, dims=[3])

                outputs = (outputs + flipped_outputs) / 2

            preds = torch.argmax(outputs, dim=1)

            for i in range(preds.size(0)):
                pred_mask = preds[i].cpu().numpy().astype(np.uint8)

                color_mask = COLOR_LUT[pred_mask]
                
                save_path = os.path.join(output_dir, img_names[i])
                Image.fromarray(color_mask).save(save_path)

    print(f"Predictions saved to: {output_dir}")
    print("Inference complete!")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()
