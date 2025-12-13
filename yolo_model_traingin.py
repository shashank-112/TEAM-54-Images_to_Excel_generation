# YOLOV8 Training Script with Epoch Weights Saving

from ultralytics import YOLO
import torch
import os

def save_epoch_callback(trainer):
    epoch = trainer.epoch
    save_dir = trainer.save_dir  # example: runs/detect/train
    weights_dir = os.path.join(save_dir, "epoch_weights")
    os.makedirs(weights_dir, exist_ok=True)

    # get current model state dict
    state_dict = trainer.model.state_dict()

    # save path
    save_path = os.path.join(weights_dir, f"epoch_{epoch}.pt")
    torch.save(state_dict, save_path)

    print(f"[INFO] Saved epoch {epoch} model to {save_path}")


if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.add_callback("on_train_epoch_end", save_epoch_callback)

    model.train(
        data="data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        workers=0,
        device=0,
        patience=20,
        cos_lr=True,
        amp=True
    )
