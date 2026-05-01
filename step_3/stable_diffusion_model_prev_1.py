# import torch
# from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel


# def load_pipeline():
#     controlnet = ControlNetModel.from_pretrained(
#         "lllyasviel/control_v11f1p_sd15_depth",
#         torch_dtype=torch.float16
#     )

#     # defining pieline
#     pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0",
#         controlnet=controlnet,
#         torch_dtype=torch.float16
#     ).to("cuda")

#     return pipe
import torch
from pathlib import Path

from diffusers import ControlNetModel, DPMSolverMultistepScheduler
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from huggingface_hub import hf_hub_download

from InstantID.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

def load_pipeline():

    # Pose ControlNet (body pose guidance)
    pose_controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        torch_dtype=torch.float16
    )

    # InstantID ControlNet (facial identity guidance)
    face_controlnet = ControlNetModel.from_pretrained(
        "InstantX/InstantID",
        subfolder="ControlNetModel",
        torch_dtype=torch.float16,
    )

    # controlnet = MultiControlNetModel([face_controlnet, pose_controlnet])
    controlnet = face_controlnet

    # InstantID SDXL Pipeline
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    # Better SDXL sampling quality for photoreal results.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )

    project_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = project_root / "InstantID" / "checkpoints"
    face_adapter_path = checkpoints_dir / "ip-adapter.bin"

    if not face_adapter_path.exists():
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ip-adapter.bin",
            local_dir=str(checkpoints_dir),
        )
        face_adapter_path = Path(downloaded_path)

    pipe.load_ip_adapter_instantid(str(face_adapter_path))

    return pipe