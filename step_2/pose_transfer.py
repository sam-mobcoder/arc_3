import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image

_POSE_PIPELINE = None

def load_pose_pipeline():
    global _POSE_PIPELINE
    if _POSE_PIPELINE is None:
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16
        )

        _POSE_PIPELINE = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

    return _POSE_PIPELINE


def apply_pose(base_image, pose_map, image_width, image_height):

    pipe = load_pose_pipeline()

    prompt = """full body photo of same person,
following exact pose,
correct anatomy,
photorealistic"""

    target_size = (image_width, image_height)

    base_image = base_image.resize(target_size)
    pose_map = pose_map.resize(target_size)
    
    
    image = pipe(
        prompt=prompt,
        image=base_image,                # THIS IS IMPORTANT
        control_image=pose_map,          # pose guidance
        strength=0.30,                   # CRITICAL (not >0.3)
        guidance_scale=2.8,
        num_inference_steps=30,
        controlnet_conditioning_scale=1.0
    ).images[0]

    return image