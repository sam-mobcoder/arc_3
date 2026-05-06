import torch
from PIL import Image

from step_3.stable_diffusion_model import load_pipeline

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
            _FACE_IMAGE_CACHE[selfie_path] = Image.open(selfie_path).convert("RGB")
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
    seed=0
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
    if pose_path is not None:
        # Pose map is currently not consumed by PuLID/FLUX generation in this file.
        # Skip expensive pose estimation to avoid unnecessary CPU-heavy work.
        pass

    # -----------------------------------
    # STEP 3 — Load Pipeline
    # -----------------------------------
    pipe = _get_pipeline()
    # -----------------------------------
    # PROMPT
    # -----------------------------------
    prompt = """upper body portrait, chest up framing, shoulders fully visible,
subject centered, camera at eye level, symmetrical composition,
same identity, same facial structure,
realistic human anatomy, correct shoulder width, natural neck connection,
    ultra realistic DSLR portrait photograph of a person,
natural human skin texture,
neutral expression,
front facing,
full face visible,
visible skin pores,
extreamly realistic and perfect skin,
high frequency facial details,
natural lighting,
real camera photography,
85mm portrait lens,
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
low quality,
do not Crop Face 
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
    )
    except Exception as exc:
        print(f"[SKIP] Generation failed for {selfie_path}: {exc}")
        return None

    return image