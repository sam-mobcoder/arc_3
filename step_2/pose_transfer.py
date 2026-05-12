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


def apply_pose(
    base_image,
    pose_path,
    pose_map,
    image_width,
    image_height,
    base_portrait=None,
    pose_init_weight=0.58,
):
    """
    Blend pose reference with Flux portrait for img2img init: pose image drives layout,
    base_portrait (PuLID output) carries identity, skin tone, and face structure.
    pose_init_weight: fraction of pose_rgb in the blend (rest is base portrait).
    """
    pipe = load_pose_pipeline()

    prompt = """professional full body photograph, exact same person as the portrait reference,
same face identity, same facial features, natural skin tone matching reference,
body pose and limb positions following OpenPose skeleton exactly,
photorealistic, sharp eyes, detailed face, natural lighting,
fully clothed, correct hands and feet, coherent anatomy"""

    negative_prompt = """different person, wrong face, identity change, duplicate face,
deformed hands, extra fingers, fused limbs, low quality, blurry face,
cartoon, illustration, doll, plastic skin, oversmoothed"""

    target_size = (image_width, image_height)

    pose_rgb = Image.open(pose_path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    pose_map = pose_map.resize(target_size, Image.Resampling.LANCZOS)

    ref = base_portrait if base_portrait is not None else base_image
    ref = ref.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)

    # PIL blend: out = pose * w_pose + ref * (1 - w_pose)
    w_pose = float(pose_init_weight)
    w_pose = max(0.35, min(0.72, w_pose))
    init_image = Image.blend(ref, pose_rgb, w_pose)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        control_image=pose_map,
        strength=0.42,
        guidance_scale=3.4,
        num_inference_steps=40,
        controlnet_conditioning_scale=1.12,
    ).images[0]

    return image