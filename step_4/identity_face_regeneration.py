'''
INPUT :- {
    scene_image
    face_embedding
    prompt
}

OUTPUT :- {
    NOthing it is orchestrator
}

'''
from PuLID.flux.sampling import (
    denoise,
    prepare,
    unpack,
    get_schedule,
)
from step_4.facial_inpainting import FacialInpainting


class IdentityFaceRegeneration:

    def __init__(
        self,
        pulid_pipeline,
    ):

        self.pipeline = FacialInpainting(
            model=pulid_pipeline.model,
            ae=pulid_pipeline.ae,
            t5=pulid_pipeline.t5,
            clip=pulid_pipeline.clip,
            pulid_model=pulid_pipeline.pulid_model,
            device=pulid_pipeline.device,
            dtype=pulid_pipeline.dtype,
            prepare=prepare,
            denoise=denoise,
            unpack=unpack,
            get_schedule=get_schedule,
        )

    def run(
        self,
        target_image,
        face_embedding,
        prompt,
        reference_portrait=None,
        uncond_id=None,
    ):

        final_image = self.pipeline.regenerate_face(
            target_image=target_image,
            id_embeddings=face_embedding,
            prompt=prompt,
            reference_portrait=reference_portrait,
            uncond_id=uncond_id,
        )

        return final_image