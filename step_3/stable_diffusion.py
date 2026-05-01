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
    ultra realistic DSLR portrait photograph of a person,
natural human skin texture,
visible skin pores,
realistic imperfect skin,
high frequency facial details,
natural lighting,
photojournalistic realism,
real camera photography,
85mm portrait lens,
shallow depth of field,
high dynamic range,
extremely detailed realistic eyes,
realistic hair strands,
RAW photograph
    """

    negative_prompt = """
    cartoon,
animation,
cgi,
3d render,
painting,
illustration,
airbrushed skin,
plastic skin,
beauty filter,
unrealistic skin,
smooth face,
fake eyes,
oversaturated,
deformed face,
blurry,
low quality
    """

    # -----------------------------------
    # GENERATION
    # -----------------------------------
    generator = torch.Generator(
        device=pipe.device.type
    )

    # -----------------------------------
    # PuLID / FLUX Generation
    # -----------------------------------
    image = pipe.generate(
        prompt=prompt,

        negative_prompt=negative_prompt,

        face_image=identity["face_pil"],

        # face_embedding=identity["embedding_tensor"],

        generator=generator,

        num_inference_steps=45,

        guidance_scale=3.0,
    )

    return image