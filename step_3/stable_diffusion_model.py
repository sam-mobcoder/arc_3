import torch

from PIL import Image

from diffusers import FluxPipeline


class PuLIDFluxPipeline:

    def __init__(self):

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def generate(
        self,
        prompt,
        negative_prompt,
        face_image,
        face_embedding,
        generator,
        num_inference_steps=30,
        guidance_scale=4.0,
    ):

        image = self.pipe(
            prompt=prompt,

            guidance_scale=guidance_scale,

            num_inference_steps=num_inference_steps,

            generator=generator,

            width=768,
            height=1024,
        ).images[0]

        return image


def load_pipeline():

    pipe = PuLIDFluxPipeline()

    return pipe