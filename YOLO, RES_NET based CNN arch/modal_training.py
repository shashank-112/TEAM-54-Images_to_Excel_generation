# model training

import os
import cv2
import torch
import numpy as np
import shutil
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )
            if in_c != out_c else nn.Identity()
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return self.relu(out)

class StrongCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ResBlock(1, 32),
            nn.MaxPool2d(2),

            ResBlock(32, 64),
            nn.MaxPool2d(2),  # becomes 8x8

            ResBlock(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 12)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = StrongCNN().to(device)
model.load_state_dict(torch.load("./model/model_epoch_190.pt", map_location=device))
model.eval()

print("✔ Model loaded successfully")

input_folder = "./crops_binary_inv"
output_root = "./sorted_digits"
os.makedirs(output_root, exist_ok=True)

for d in range(12):
    os.makedirs(os.path.join(output_root, str(d)), exist_ok=True)

for fname in os.listdir(input_folder):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue

    path = os.path.join(input_folder, fname)
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0

    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img).argmax(dim=1).item()

    shutil.copy(path, os.path.join(output_root, str(pred), fname))

print("\n✔ All predictions completed and images sorted into folders.")

# model accuracy: 98.75% on test set and this is a resnet-based CNN model with 3 residual blocks