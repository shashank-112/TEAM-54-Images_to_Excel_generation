from pdf2image import convert_from_path
import cv2
import numpy as np
import os
import torch
import torch.nn as nn

POPPLER_PATH = r'C:\Program Files\poppler-24.08.0\Library\bin'
OUTPUT_FOLDER = "./all_processed_images/T1_all_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class PdfToImageConverter:
    def __init__(self, path):
        self.pdf_path = path
        
    def execute(self):
        self.convert_Pdf_To_Images()
        self.ppl_To_OpenCV()

        self.count_No_Of_Pages()
        self.save_CV2_Type_Images()
        
        return self.cv2_type_images, self.total_no_of_pages
        
    def convert_Pdf_To_Images(self):
        self.individual_images = convert_from_path(
                                    self.pdf_path, 
                                    dpi = 110,
                                    poppler_path = POPPLER_PATH
                                )
        
    def ppl_To_OpenCV(self):
        self.cv2_type_images = []
        
        for i, img in enumerate(self.individual_images):
            np_type_img = np.array(img)
            cv2_type_img = cv2.cvtColor(np_type_img, cv2.COLOR_RGB2BGR)
            
            self.cv2_type_images.append(cv2_type_img)
        
    def count_No_Of_Pages(self):
        self.total_no_of_pages = len(self.cv2_type_images)
            
    def save_CV2_Type_Images(self):
        for i, img in enumerate(self.cv2_type_images):
            cv2.imwrite(f"{OUTPUT_FOLDER}/{i}.jpg", img)
            
            
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
        self.shortcut = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        return self.relu(out)

class StrongCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ResBlock(1, 32),
            nn.MaxPool2d(2),
            
            ResBlock(32, 64),
            nn.MaxPool2d(2),
            
            ResBlock(64, 128)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            
            nn.Dropout(0.25),
            nn.Linear(512, 12)
        )

    def forward(self, x):
        return self.classifier(self.features(x))