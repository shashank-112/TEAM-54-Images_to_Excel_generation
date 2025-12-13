# bits croper

from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("./model/best.pt")

def sort_boxes_xyxy(boxes):
    boxes = np.array(boxes)
    sorted_by_y = boxes[boxes[:, 1].argsort()]
    row1 = sorted_by_y[:17]
    row2 = sorted_by_y[17:]
    row1 = row1[row1[:, 0].argsort()]
    row2 = row2[row2[:, 0].argsort()]
    return np.vstack((row1, row2))


os.makedirs("./crops", exist_ok=True)

for i in range(1, 106):

    image_path = f"./images/{i}.jpg"
    results = model.predict(image_path, conf=0.45)[0]
    img = cv2.imread(image_path)

    padding = 0

    all_boxes = []
    all_confs = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf.cpu().numpy())
        all_boxes.append([x1, y1, x2, y2])
        all_confs.append(conf)

    all_boxes = np.array(all_boxes)
    all_confs = np.array(all_confs)

    # pick top 34 by confidence
    keep_idx = np.argsort(-all_confs)[:34]
    boxes_34 = all_boxes[keep_idx]

    # sort into 2 rows and left→right
    sorted_boxes = sort_boxes_xyxy(boxes_34)

    # remove boxes that are too close
    filtered_boxes = []
    last_x = None
    for box in sorted_boxes:
        x1, y1, x2, y2 = box
        if last_x is None:
            filtered_boxes.append(box)
            last_x = x1
        else:
            if abs(x1 - last_x) >= 20:
                filtered_boxes.append(box)
                last_x = x1

    filtered_boxes = np.array(filtered_boxes)

    # ---- CROP AND SAVE ----
    box_id = 1
    for (x1, y1, x2, y2) in filtered_boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # apply padding safely
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(img.shape[1], x2 + padding)
        py2 = min(img.shape[0], y2 + padding)
        margin = 19
        crop = img[py1+margin:py2-margin, px1+margin:px2-margin]

        out_path = f"./crops/{i}-{box_id}.jpg"
        cv2.imwrite(out_path, crop)
        box_id += 1

    print("done", i)
