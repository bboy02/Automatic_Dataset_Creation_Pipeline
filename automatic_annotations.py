import os
from torch import nn
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from torchvision.ops import box_convert
from PIL import Image
from pathlib import Path

# Helper: Overlay mask on image
def show_mask(mask, image, color=(0, 255, 0), alpha=0.5):
    mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color)
    return overlay.astype(np.uint8)

# Helper: Convert box from xyxy to YOLOv8 format (class x_center y_center width height)
def convert_to_yolov8(box_xyxy, img_w, img_h, class_id=0):
    x_min, y_min, x_max, y_max = box_xyxy
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# Main Function
def detect_and_segment(image_path, text_prompt, class_id=0, output_dir="output", box_threshold=0.35, text_threshold=0.25):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Load GroundingDINO model
    model = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )

    # Load image
    image_source, image = load_image(image_path)

    # Detect bounding boxes with GroundingDINO
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    if len(boxes) == 0:
        print(f"No detection in {image_path}")
        return

    # Annotate image with boxes
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # Load SAM model and predictor
    sam_checkpoint = "/home/student/pc_deploy/Semester_2/BikeSafeAI/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    # Set image for SAM
    image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)

    # Convert boxes for SAM
    H, W, _ = image_source.shape
    boxes = boxes.to(device)
    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    scale = torch.tensor([W, H, W, H], dtype=torch.float32, device=device)
    boxes_xyxy = boxes_xyxy * scale
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)

    # Get masks
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks.cpu().numpy()

    # Draw masks
    segmented_frame = annotated_frame.copy()
    for i in range(len(boxes)):
        segmented_frame = show_mask(masks[i][0], segmented_frame)

    # Save segmented image
    img_name = Path(image_path).stem
    out_image_path = os.path.join(output_dir, "images", f"{img_name}.jpg")
    out_label_path = os.path.join(output_dir, "labels", f"{img_name}.txt")
    Image.fromarray(segmented_frame).save(out_image_path)

    # Save YOLOv8 label file
    with open(out_label_path, "w") as f:
        for box in boxes_xyxy.cpu().numpy():
            yolo_line = convert_to_yolov8(box, W, H, class_id=class_id)
            f.write(yolo_line + "\n")

    print(f"✅ Processed: {image_path}")


# Folder-to-class mapping
category_map = {
    "cycles": 2
}

base_path = "Dataset_new"

for category, class_id in category_map.items():
    folder_path = os.path.join(base_path, category)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            try:
                detect_and_segment(
                    image_path=image_path,
                    text_prompt="cycle",
                    class_id=class_id,
                    output_dir="yolo_annotations1"
                )
            except Exception as e:
                print(f"Error in {filename}: {e}")

