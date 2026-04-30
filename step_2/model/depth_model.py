import torch
from transformers import AutoImageProcessor, DPTForDepthEstimation

def load_depth_model():
    feature_extractor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    return feature_extractor, model