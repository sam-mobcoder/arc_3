
from PIL import Image
import numpy as np
import torch
from .model.pose_model import load_pose_model
from .model.depth_model import load_depth_model
from controlnet_aux import OpenposeDetector

_POSE_DETECTOR = None

def _get_pose_detector():
    global _POSE_DETECTOR
    if _POSE_DETECTOR is None:
        _POSE_DETECTOR = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    return _POSE_DETECTOR

def get_pose_estimation(
    pose_path, 
    depth=False
):

    pose_detector = _get_pose_detector()

    # open image
    pose_image = Image.open(pose_path).convert("RGB")
    pose_image = np.array(pose_image)
    
    pose_map = pose_detector(pose_image)
    # pose_image = Image.open(pose_image_path).convert("RGB")
    # openpose = load_pose_model()
    # pose_map = openpose(pose_image)

    # depth_map = None
    # if depth:
    #     feature_extractor, model = load_depth_model()
    #     inputs = feature_extractor(pose_image, return_tensors="pt").to(torch.device("cuda"))
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #     depth_map = outputs.predicted_depth

    # return {
    #     "pose_map": pose_map,
    #     # "depth_map": depth_map
    # }
    return pose_map
