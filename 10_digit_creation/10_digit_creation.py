# the reason we are doing this is because our data there is a chance that there 
# will be a 10 digit number so we need to train the model for that as well

import cv2
import numpy as np

# ---------------------------------
# PATHS
# ---------------------------------
IMG_1_PATH = "./images/1.png"   # digit 1 (28x28)
IMG_0_PATH = "./images/2.png"   # digit 0 (28x28)
OUTPUT_PATH = "./images/combined_10.png"

# ---------------------------------
# LOAD IMAGES (GRAYSCALE)
# ---------------------------------
img1 = cv2.imread(IMG_1_PATH, cv2.IMREAD_GRAYSCALE)
img0 = cv2.imread(IMG_0_PATH, cv2.IMREAD_GRAYSCALE)

if img1 is None or img0 is None:
    raise ValueError("Image path incorrect")

# ---------------------------------
# FUNCTION: THRESHOLD + CONTOUR CROP
# ---------------------------------
def contour_crop(img):
    # 1. Threshold (digit -> white, background -> black)
    _, thresh = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 2. Find contours on thresholded image
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 3. Select largest contour (digit)
    c = max(contours, key=cv2.contourArea)

    # 4. Get edge-based bounding box
    x, y, w, h = cv2.boundingRect(c)

    # 5. Tight crop (no padding)
    cropped = img[y:y+h, x:x+w]

    return cropped

# ---------------------------------
# CROP BOTH DIGITS
# ---------------------------------
crop1 = contour_crop(img1)
crop0 = contour_crop(img0)

# ---------------------------------
# RESIZE TO SAME HEIGHT (KEEP RATIO)
# ---------------------------------
def resize_keep_ratio(img, target_h):
    h, w = img.shape
    new_w = int(w * target_h / h)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

target_height = max(crop1.shape[0], crop0.shape[0])

crop1 = resize_keep_ratio(crop1, target_height)
crop0 = resize_keep_ratio(crop0, target_height)

combined = np.hstack((crop1, crop0))

final_img = cv2.resize(
    combined,
    (28, 28),
    interpolation=cv2.INTER_AREA
)

cv2.imwrite(OUTPUT_PATH, final_img)

cv2.imshow("Cropped 1", crop1)
cv2.imshow("Cropped 0", crop0)
cv2.imshow("Final 10 (28x28)", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Saved:", OUTPUT_PATH)
