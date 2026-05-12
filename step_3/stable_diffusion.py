import torch
import numpy as np
from PIL import Image

from step_2.pose_transfer import apply_pose
from step_2.pose_estimation import get_pose_estimation
from step_3.stable_diffusion_model import load_pipeline
from step_4.identity_face_regeneration import ( IdentityFaceRegeneration )

_PIPELINE = None
_FACE_IMAGE_CACHE = {}


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = load_pipeline()
    return _PIPELINE


def _get_face_image(selfie_path: str):
    """
    Keep preprocessing lightweight on CPU and let PuLID pipeline do the heavy
    identity extraction on GPU/ONNX-CUDA.
    """
    if selfie_path not in _FACE_IMAGE_CACHE:
        try:
            _FACE_IMAGE_CACHE[selfie_path] = Image.open(selfie_path).convert("RGB").resize((512, 512))
        except Exception as exc:
            print(f"[SKIP] Failed to read image: {selfie_path} ({exc})")
            _FACE_IMAGE_CACHE[selfie_path] = None
    return _FACE_IMAGE_CACHE[selfie_path]


def generate_image(
    selfie_path,
    pose_path=None,
    num_inference_steps=0,
    guidance_scale=0,
    id_weight=0,
    seed=0,
    image_width=832,
    image_height=1216,
):

    # -----------------------------------
    # STEP 1 — Identity
    # -----------------------------------
    face_image = _get_face_image(selfie_path)
    
    if face_image is None:
        print(f"[SKIP] Invalid source image: {selfie_path}")
        return None

    """
    identity = {
        "face_pil": ...,
        "embedding_tensor": ...
    }
    """

    # -----------------------------------
    # STEP 2 — Pose (optional later)
    # -----------------------------------
    # if pose_path is not None:
    #     # Pose map is currently not consumed by PuLID/FLUX generation in this file.
    #     # Skip expensive pose estimation to avoid unnecessary CPU-heavy work.
    #     pass

    # -----------------------------------
    # STEP 3 — Load Pipeline
    # -----------------------------------
    pipe = _get_pipeline()
    identity_regenerator = IdentityFaceRegeneration(
        pulid_pipeline=pipe,
    )

    selfie_id_embeddings, uncond_id_embeddings = (
        pipe.pulid_model.get_id_embedding(
            np.array(face_image),
            cal_uncond=True,
        )
    )

    # -----------------------------------
    # PROMPT
    # -----------------------------------
    prompt = """
    mid body portrait, waist up,
hands partially visible,
same person, same identity,

wearing casual t-shirt,
fully clothed,

natural human anatomy,
correct shoulder width,
natural neck connection,

extremely detailed face,
visible skin pores,
ultra realistic eyes,
realistic skin texture,

photorealistic,
natural lighting,
85mm lens,
DSLR photo
neutral lips,
natural mouth shape,
minimal makeup,
natural female face
    """
#     prompt = """upper body portrait, chest up framing, shoulders fully visible,
# subject centered, camera at eye level, symmetrical composition,
# same identity, same facial structure,
# realistic human anatomy, correct shoulder width, natural neck connection,
#     ultra realistic DSLR portrait photograph of a person,
# natural human skin texture,
# neutral expression,
# front facing,
# full face visible,
# visible skin pores,
# extreamly realistic and perfect skin,
# high frequency facial details,
# natural lighting,
# real camera photography,
# 85mm portrait lens,
# high dynamic range,
# extremely detailed realistic eyes,
# realistic hair strands,
# RAW photograph
#     """


   
    negative_prompt = """
    nude, naked, topless, bare chest, exposed breasts,
    nsfw, erotic, cleavage,
    sexual, explicit, provocative,
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
    low quality,
    do not Crop Face,
    makeup,
    lipstick,
    glossy lips,
    pursed lips,
    duck face,
    beauty model face,
    fashion makeup,
    heavy eyeliner
    """

    # -----------------------------------
    # GENERATION
    # -----------------------------------
    generator = torch.Generator(
        device=pipe.device.type
    )

    # getting pose 

    # -----------------------------------
    # PuLID / FLUX Generation
    # -----------------------------------
    try:
        image = pipe.generate(
            prompt=prompt,

            negative_prompt=negative_prompt,
            face_image=face_image,
            # face_embedding=identity["embedding_tensor"],

            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            id_weight=id_weight,
            width=image_width,
            height=image_height,
        )
        with open("base_image.png", "wb") as f:
            image.save(f)

        pose_map = get_pose_estimation(pose_path)

        final_image = apply_pose(image, pose_path, pose_map, image_width, image_height)

        print("-----------> Identity Face Regeneration:-")
        try:
            final_image = identity_regenerator.run(
                target_image=final_image,
                face_embedding=selfie_id_embeddings,
                prompt=prompt,
                reference_portrait=image,
                uncond_id=uncond_id_embeddings,
            )
        except Exception as reg_exc:
            print(f"[WARN] Step 4 face regeneration skipped, using pose-stage image only: {reg_exc}")

        print("-----------> Identity Face Regeneration Complete:-")

    except Exception as exc:
        print(f"[SKIP] Generation failed for {selfie_path}: {exc}")
        return None

    return final_image