from controlnet_aux import OpenposeDetector

def load_pose_model():
    return OpenposeDetector.from_pretrained("lllyasviel/ControlNet")