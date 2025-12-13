# image saturation with thresholding(THRESH_BINARY_INV)

import cv2
import numpy as np
import glob
import os

input_folder = "./crops/"
binary_output_folder = "./crops_binary_inv/"

os.makedirs(binary_output_folder, exist_ok=True)

lower_red1 = np.array([0, 15, 15])
upper_red1 = np.array([25, 255, 255])

lower_red2 = np.array([155, 15, 15])
upper_red2 = np.array([179, 255, 255])

kernel = np.ones((13,13), np.uint8)

for img_path in glob.glob(input_folder + "/*.jpg"):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    h, s, v = cv2.split(hsv)

    s = s.astype(np.float32)
    s[mask > 0] *= 4
    s = np.clip(s, 0, 255).astype(np.uint8)

    hsv_enhanced = cv2.merge([h, s, v])
    final_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    filename = os.path.basename(img_path)

    gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    binary_inv = cv2.resize(binary_inv, (32, 32), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(binary_output_folder, filename), binary_inv)

    print("Saved:", filename, "and binary_inv version")