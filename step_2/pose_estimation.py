
from PIL import Image
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

from .model.pose_model import load_pose_model
from .model.depth_model import load_depth_model


def get_pose_estimation(
    pose_image_path, 
    depth=False
):
    pose_image = Image.open(pose_image_path).convert("RGB")
    openpose = load_pose_model()
    pose_map = openpose(pose_image)

    depth_map = None
    if depth:
        feature_extractor, model = load_depth_model()
        inputs = feature_extractor(pose_image, return_tensors="pt").to(torch.device("cuda"))
        with torch.no_grad():
            outputs = model(**inputs)
        depth_map = outputs.predicted_depth

    return {
        "pose_map": pose_map,
        "depth_map": depth_map
    }

