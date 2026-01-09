"""
This script implements standalone inference on your images using the trained MassID45 Mask-RCNN model. 
We DO NOT perform tiled inference here. See `sahi_inference.sh` to perform tiled inference on large images
"""
import logging
import argparse
import os
import glob
import time
from typing import List, Optional

import numpy as np
import torch
import cv2

# Detectron2 imports
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.data.transforms as T

# SAHI imports (kept from your snippet, though not strictly used in standard D2 inference)
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model(model_path, config_path):
    """
    Loads the model and config once to avoid overhead during directory iteration.
    """
    logger.info(f"Loading config from {config_path}...")
    cfg = LazyConfig.load(config_path)
    cfg.train.init_checkpoint = model_path
    cfg.train.device = str(device)

    # Instantiate model
    logger.info(f"Loading model weights from {model_path}...")
    model = instantiate(cfg.model)
    DetectionCheckpointer(model).load(model_path)
    model.to(device)
    model.eval()
    
    return model, cfg

def predict_on_single_image(model, cfg, img_path, debug=False):
    """
    Runs inference on a single image using the pre-loaded model.
    """
    # Read image using OpenCV (BGR format)
    original_image = cv2.imread(img_path)
    if original_image is None:
        logger.error(f"Could not read image: {img_path}")
        return None, None

    # Get ORIGINAL dimensions
    orig_height, orig_width = original_image.shape[:2]

    # Preprocessing Logic
    # 1. Convert to RGB for the transform
    image = original_image[:, :, ::-1]
    
    # 2. Apply Transforms defined in the LazyConfig mapper
    mapper = instantiate(cfg.dataloader.test.mapper)
    aug = mapper.augmentations
    # AugInput requires RGB
    aug_input = T.AugInput(image)
    transformed_image = aug(aug_input).apply_image(image)
    
    # 3. Convert to Tensor and move to device
    image_tensor = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
    image_tensor = image_tensor.to(device)

    # 4. Run Inference
    with torch.no_grad():
        # CRITICAL FIX: Pass 'orig_height' and 'orig_width' here.
        # This tells the model: "The tensor is 1024x1024, but please scale 
        # the output predictions back to orig_height x orig_width."
        inputs = {"image": image_tensor, "height": orig_height, "width": orig_width}
        
        # Model expects a list of dicts
        prediction_result = model([inputs])[0]

    if debug:
        print("\n--- DEBUG INFO ---")
        instances = prediction_result['instances']
        print(f"Detected {len(instances)} objects")
        if len(instances) > 0:
            if instances.has("pred_masks"):
                sample_preds = instances.pred_masks.cpu().detach().numpy()
                print(f"Mask Shape: {sample_preds.shape} (Should match original image HxW)")

    return prediction_result, original_image

def display_predictions(image, predictions, output_path, metadata):
    """
    Visualizes predictions and saves the result to disk.
    """
    if predictions is None:
        return

    # Convert BGR (OpenCV) to RGB for Visualizer
    visualizer = Visualizer(
        image[:, :, ::-1], 
        metadata=metadata, 
        scale=1.0, 
        instance_mode=ColorMode.IMAGE
    )
    
    # Draw predictions
    # We move instances to CPU before visualization
    instances = predictions["instances"].to("cpu")
    vis_output = visualizer.draw_instance_predictions(predictions=instances)

    # Get the result image (in RGB) and convert back to BGR for OpenCV saving
    result_image = vis_output.get_image()[:, :, ::-1]

    # Save
    cv2.imwrite(output_path, result_image)
    logger.info(f"Saved visualization to: {output_path}")

def main(args):
    # 1. Setup Configuration and Model
    # Note: If your config path is hardcoded in your project structure, you can keep the default string below
    default_config = "configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py"
    config_path = args.config if args.config else default_config
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    model, cfg = setup_model(args.model_path, config_path)

    # 2. Setup Metadata for Visualization
    # Define your specific mapping here
    category_mapping = {"1": "b"} 
    # Create a dummy metadata catalog for visualization
    metadata = MetadataCatalog.get("massid45_inference")
    metadata.thing_classes = list(category_mapping.values())
    # Note: If your model outputs class ID 0, ensure 'b' is the 0th element in this list.
    # If the model was trained on more classes, this list must match the training class order.

    # 3. Handle Input Directory or File
    input_path = args.imgs_directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    if os.path.isdir(input_path):
        # Grab common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
    elif os.path.isfile(input_path):
        image_files = [input_path]
    else:
        logger.error(f"Input path {input_path} is not valid.")
        return

    logger.info(f"Found {len(image_files)} images to process.")

    # 4. Run Inference Loop
    for i, img_file in enumerate(image_files):
        filename = os.path.basename(img_file)
        save_path = os.path.join(output_dir, f"pred_{filename}")

        logger.info(f"[{i+1}/{len(image_files)}] Processing {filename}...")
        
        predictions, original_img = predict_on_single_image(
            model, 
            cfg, 
            img_file, 
            debug=args.debug
        )

        if predictions is not None:
            display_predictions(original_img, predictions, save_path, metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone inference script for MassID45 images")
    
    # Path Arguments
    parser.add_argument("--model_path", required=True, help="Path to the model .pth file")
    parser.add_argument("--imgs_directory", required=True, help="Path to input image or directory of images")
    parser.add_argument("--output_dir", default="output_predictions", help="Directory to save visualized outputs")
    
    # Optional Arguments
    parser.add_argument("--config", type=str, default=None, help="Path to the LazyConfig .py file (optional if hardcoded)")
    parser.add_argument("--debug", action="store_true", help="Print debug information about masks/boxes")
    
    args = parser.parse_args()
    
    main(args)