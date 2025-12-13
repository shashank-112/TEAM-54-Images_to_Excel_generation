# image saturation

import cv2
import numpy as np
import glob
import os

input_folder = "./crops/"
output_folder = "./crops_output/"
os.makedirs(output_folder, exist_ok=True)

lower_red1 = np.array([0, 15, 15])
upper_red1 = np.array([25, 255, 255])

lower_red2 = np.array([155, 15, 15])
upper_red2 = np.array([179, 255, 255])

kernel = np.ones((3,3), np.uint8)

for img_path in glob.glob(input_folder + "/*.jpg"):  # change to jpg if needed
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
    cv2.imwrite(os.path.join(output_folder, filename), final_img)

    print("Saved:", filename)