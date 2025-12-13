# image generator with out box but adding noise and augmenting images

import cv2
import numpy as np
import os
import random

INPUT_DIR = "./normal data"
OUTPUT_DIR = "./augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- AUGMENTATION FUNCTIONS ----------

def random_translate(img):
    h, w = img.shape
    tx = random.randint(-7, 7)
    ty = random.randint(-7, 7)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderValue=0)

def add_noise(img):
    choice = random.choice(["gaussian", "saltpepper", "none"])
    img = img.copy()

    if choice == "gaussian":
        noise = np.random.normal(0, 20, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif choice == "saltpepper":
        prob = 0.02
        rnd = np.random.rand(*img.shape)
        img[rnd < prob] = 0
        img[rnd > 1 - prob] = 255

    return img

def random_scale(img):
    scale = random.uniform(0.85, 1.15)
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros_like(img)
    h, w = img.shape
    rh, rw = resized.shape

    y = (h - rh) // 2
    x = (w - rw) // 2

    if y >= 0 and x >= 0:
        canvas[y:y+rh, x:x+rw] = resized[:h-y, :w-x]
    else:
        canvas = cv2.resize(resized, (w, h), interpolation=cv2.INTER_NEAREST)

    return canvas

def random_rotate(img):
    angle = random.uniform(-10, 10)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=0)

def random_margin_crop(img):
    # crop and pad simulated
    top = random.randint(0, 3)
    bottom = random.randint(0, 3)
    left = random.randint(0, 3)
    right = random.randint(0, 3)

    cropped = img[top:28-bottom, left:28-right]

    padded = cv2.copyMakeBorder(
        cropped,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=0
    )
    return padded

# ---- EXTRA: tiny strokes for empty class ----

def random_stray_stroke():
    img = np.zeros((28, 28), dtype=np.uint8)

    # 70% chance no stroke (empty)
    if random.random() < 0.7:
        return img

    # 30% chance: generate stroke
    length = random.randint(3, 9)
    thickness = random.randint(1, 3)
    angle = random.uniform(-40, 40)  # roughly horizontal

    # pick LEFT or RIGHT side
    if random.random() < 0.5:
        # LEFT SIDE
        x1 = random.randint(0, 5)
    else:
        # RIGHT SIDE
        x1 = random.randint(22, 27)

    y1 = random.randint(5, 22)  # avoid top/bottom

    # compute endpoint from angle + length
    dx = int(length * np.cos(np.radians(angle)))
    dy = int(length * np.sin(np.radians(angle)))

    x2 = x1 + dx
    y2 = y1 + dy

    # clamp endpoints
    x2 = np.clip(x2, 0, 27)
    y2 = np.clip(y2, 0, 27)

    cv2.line(img, (x1, y1), (x2, y2), color=255, thickness=thickness)

    return img


# ---------- MAIN PROCESS ----------

for cls in range(0, 12):  # 0–11
    in_path = os.path.join(INPUT_DIR, str(cls))
    out_path = os.path.join(OUTPUT_DIR, str(cls))
    os.makedirs(out_path, exist_ok=True)

    file_list = os.listdir(in_path)

    if cls == 11:
        # empty class special handling
        for i in range(5000):  # generate 5k empty samples
            img = random_stray_stroke()
            cv2.imwrite(os.path.join(out_path, f"{i}.png"), img)
        print("Generated empty-class augmented images.")
        continue

    # normal digit classes 0–10
    for fname in file_list:
        img = cv2.imread(os.path.join(in_path, fname), cv2.IMREAD_GRAYSCALE)

        for k in range(4):  # generate 4 augmented copies per image
            aug = img.copy()
            aug = random_translate(aug)
            aug = random_scale(aug)
            aug = random_rotate(aug)
            aug = random_margin_crop(aug)
            aug = add_noise(aug)

            save_name = fname.replace(".png", "").replace(".jpg", "")
            cv2.imwrite(os.path.join(out_path, f"{save_name}_aug{k}.png"), aug)

    print("Done class:", cls)