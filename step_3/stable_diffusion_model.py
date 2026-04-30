import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel


def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16
    )

    # defining pieline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    return pipe