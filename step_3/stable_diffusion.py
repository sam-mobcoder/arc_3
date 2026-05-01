import torch

from step_1.pulid_flux import prepare_pulid_data

from step_2.pose_estimation import get_pose_estimation

from step_3.stable_diffusion_model import load_pipeline


def generate_image(
    selfie_path,
    pose_path=None,
):

    # -----------------------------------
    # STEP 1 — Identity
    # -----------------------------------
    identity = prepare_pulid_data(
        selfie_path
    )

    """
    identity = {
        "face_pil": ...,
        "embedding_tensor": ...
    }
    """

    # -----------------------------------
    # STEP 2 — Pose (optional later)
    # -----------------------------------
    pose_map = None

    if pose_path is not None:
        pose_map = get_pose_estimation(
            pose_path
        )

    # -----------------------------------
    # STEP 3 — Load Pipeline
    # -----------------------------------
    pipe = load_pipeline()

    # -----------------------------------
    # PROMPT
    # -----------------------------------
    prompt = """
    photorealistic portrait photo of a person,
    ultra realistic skin texture,
    realistic eyes,
    natural lighting,
    highly detailed face,
    realistic photography
    """

    negative_prompt = """
    blurry,
    distorted face,
    deformed anatomy,
    cartoon,
    painting,
    illustration,
    low quality,
    unrealistic eyes
    """

    # -----------------------------------
    # GENERATION
    # -----------------------------------
    generator = torch.Generator(
        device="cuda"
    ).manual_seed(123456)

    # -----------------------------------
    # PuLID / FLUX Generation
    # -----------------------------------
    image = pipe.generate(
        prompt=prompt,

        negative_prompt=negative_prompt,

        face_image=identity["face_pil"],

        face_embedding=identity["embedding_tensor"],

        generator=generator,

        num_inference_steps=30,

        guidance_scale=4.0,
    )

    return image