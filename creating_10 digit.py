# digit 10 generation

import cv2
import numpy as np
import os
import random

def get_digit_crop(img):
    """Extract the digit using contour detection and return cropped digit."""
    gray = img.copy()

    # threshold
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # biggest contour (digit region)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return gray[y:y+h, x:x+w]


def load_random_digit(folder):
    """Load a random image from a folder and return its cropped digit."""
    filenames = os.listdir(folder)
    fname = random.choice(filenames)
    img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
    return get_digit_crop(img)


def create_digit_10():
    """Create a single synthetic '10' digit image."""

    # --- Load digits ---
    d1 = load_random_digit("1")
    d0 = load_random_digit("0")

    if d1 is None or d0 is None:
        return None

    # --- Random spacing between 2–8 px ---
    spacing = random.randint(2, 8)
    space = np.zeros((max(d1.shape[0], d0.shape[0]), spacing), dtype=np.uint8)

    # --- Combine horizontally ---
    h = max(d1.shape[0], d0.shape[0])
    d1_pad = cv2.copyMakeBorder(d1, 0, h - d1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    d0_pad = cv2.copyMakeBorder(d0, 0, h - d0.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)

    combined = np.hstack((d1_pad, space, d0_pad))

    # --- Resize to 28x28 ---
    img = cv2.resize(combined, (28, 28), interpolation=cv2.INTER_AREA)

    # --- Add random padding: 1–8 pix each side ---
    t = random.randint(1, 8)
    b = random.randint(1, 8)
    l = random.randint(1, 8)
    r = random.randint(1, 8)

    padded = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

    # resize again to 28x28 final size
    final_img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    return final_img


# ============================
# Generate 10 images
# ============================
os.makedirs("10", exist_ok=True)

for i in range(10):
    img10 = create_digit_10()
    if img10 is not None:
        cv2.imwrite(f"10/{i}.png", img10)

print("Done! 10 synthetic '10' digits created in folder '10'.")
