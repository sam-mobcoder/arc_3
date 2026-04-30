from .model.mediapipe_model import get_body_features
from PIL import Image


def get_body_estimation(
    image_path, bbox, landmarks
):
    image = Image.open(image_path).convert("RGB")
    return get_body_features(image, bbox, landmarks)