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


def apply_pose(base_image, pose_path, pose_map, image_width, image_height):
    """
    Use the pose reference photo as img2img init so layout follows the pose image;
    OpenPose map aligns skeleton. base_image kept for API compatibility.
    """
    pipe = load_pose_pipeline()

    prompt = """full body photograph, same identity as reference portrait,
exact pose and body layout as OpenPose control,
photorealistic, natural skin, detailed face,
correct anatomy"""

    target_size = (image_width, image_height)

    pose_rgb = Image.open(pose_path).convert("RGB").resize(target_size)
    pose_map = pose_map.resize(target_size)

    image = pipe(
        prompt=prompt,
        image=pose_rgb,
        control_image=pose_map,
        strength=0.48,
        guidance_scale=3.2,
        num_inference_steps=35,
        controlnet_conditioning_scale=1.15,
    ).images[0]

    return image