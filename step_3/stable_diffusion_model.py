# import torch

# from PIL import Image

# from diffusers import FluxPipeline


# class PuLIDFluxPipeline:

#     def __init__(self):

#         self.pipe = FluxPipeline.from_pretrained(
#             "black-forest-labs/FLUX.1-dev",
#             torch_dtype=torch.bfloat16
#         ).to("cuda")

#     def generate(
#         self,
#         prompt,
#         negative_prompt,
#         face_image,
#         face_embedding,
#         generator,
#         num_inference_steps=30,
#         guidance_scale=4.0,
#     ):

#         image = self.pipe(
#             prompt=prompt,

#             guidance_scale=guidance_scale,

#             num_inference_steps=num_inference_steps,

#             generator=generator,

#             width=768,
#             height=1024,
#         ).images[0]

#         return image


# def load_pipeline():

#     pipe = PuLIDFluxPipeline()

#     return pipe


# ----------------------------------- Above is for Demo -----------------------------------

import torch
import numpy as np
from contextlib import nullcontext

from PuLID.pulid.pipeline_flux import PuLIDPipeline

from PuLID.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)

from PuLID.flux.sampling import (
    denoise,
    get_noise,
    get_schedule,
    prepare,
    unpack,
)

from einops import rearrange

from PIL import Image


def resolve_runtime_device() -> torch.device:
    """
    Prefer CUDA only when the GPU arch is supported by this PyTorch build.
    Falls back to CPU for unsupported/newer GPUs (e.g. sm_120 mismatch).
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
        if arch in supported:
            return torch.device("cuda")
    except Exception:
        pass

    return torch.device("cpu")


class PuLIDFluxPipeline:

    def __init__(self):

        self.device = resolve_runtime_device()
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # -----------------------------------
        # LOAD FLUX MODELS
        # -----------------------------------
        self.t5 = load_t5(
            self.device,
            max_length=128
        )

        self.clip = load_clip(
            self.device
        )

        self.model = load_flow_model(
            "flux-dev",
            device=self.device
        )

        self.ae = load_ae(
            "flux-dev",
            device=self.device
        )

        # -----------------------------------
        # LOAD PULID
        # -----------------------------------
        self.pulid_model = PuLIDPipeline(
            self.model,
            device=self.device,
            weight_dtype=self.dtype,
        )

        self.pulid_model.load_pretrain(
            "models/pulid_flux_v0.9.1.safetensors",
            version="v0.9.1"
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        negative_prompt,
        face_image,
        generator,
        width=768,
        height=1024,
        num_inference_steps=28,
        guidance_scale=4.0,
        id_weight=0.9,
    ):

        # -----------------------------------
        # NOISE
        # -----------------------------------
        seed = torch.seed()

        x = get_noise(
            1,
            height,
            width,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )

        timesteps = get_schedule(
            num_inference_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        # -----------------------------------
        # PROMPT
        # -----------------------------------
        inp = prepare(
            t5=self.t5,
            clip=self.clip,
            img=x,
            prompt=prompt,
        )

        # -----------------------------------
        # FACE IMAGE -> NUMPY
        # -----------------------------------
        id_image = np.array(face_image)

        # -----------------------------------
        # PULID IDENTITY
        # -----------------------------------
        id_embeddings, uncond_id_embeddings = (
            self.pulid_model.get_id_embedding(
                id_image,
                cal_uncond=False
            )
        )

        # -----------------------------------
        # DENOISE
        # -----------------------------------
        x = denoise(
            self.model,

            **inp,

            timesteps=timesteps,

            guidance=guidance_scale,

            id=id_embeddings,

            id_weight=id_weight,

            start_step=0,

            uncond_id=uncond_id_embeddings,

            true_cfg=1.0,
        )

        # -----------------------------------
        # DECODE
        # -----------------------------------
        x = unpack(
            x.float(),
            height,
            width
        )

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with autocast_ctx:
            x = self.ae.decode(x)

        x = x.clamp(-1, 1)

        x = rearrange(
            x[0],
            "c h w -> h w c"
        )

        img = Image.fromarray(
            (
                127.5 * (x + 1.0)
            ).cpu().byte().numpy()
        )

        return img


def load_pipeline():

    return PuLIDFluxPipeline()