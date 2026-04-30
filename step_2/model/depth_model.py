import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

def load_depth_model():
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    return feature_extractor, model