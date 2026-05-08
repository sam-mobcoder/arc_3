# FILE: step_3/pulid_pose_transfer.py
import sys

sys.path.append("/root/arc_3/IP-Adapter")

import torch
import cv2

from PIL import Image

from diffusers import (
    StableDiffusionXLImg2ImgPipeline
)

from insightface.app import FaceAnalysis

from ip_adapter.ip_adapter_faceid import (
    IPAdapterFaceIDPlusXL
)


# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda"

# =====================================================
# LOAD SDXL
# =====================================================

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to(DEVICE)

pipe.enable_xformers_memory_efficient_attention()

# =====================================================
# LOAD INSIGHTFACE
# =====================================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=['CUDAExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

# =====================================================
# LOAD IP-ADAPTER
# =====================================================

ip_model = IPAdapterFaceIDPlusXL(
    pipe,
    "/root/arc_3/models/image_encoder/IP-Adapter/models/image_encoder",
    "/root/arc_3/models/ip-adapter-faceid-plusv2_sdxl.bin",
    DEVICE
)

# =====================================================
# MAIN FUNCTION
# =====================================================

@torch.inference_mode()
def generate_identity_transfer(
    selfie_path,
    pose_path,
    output_path="final_output.png"
):

    # ============================================
    # LOAD SELFIE
    # ============================================

    selfie = cv2.imread(
        selfie_path
    )

    print("SELFIE PATH:", selfie_path)

    if selfie is None:
        raise Exception("Image not loaded")
    
    rgb = cv2.cvtColor(
        selfie,
        cv2.COLOR_BGR2RGB
    )
    face_image = Image.fromarray(rgb)

    faces = app.get(rgb)

    print("FACES FOUND:", len(faces))

    if len(faces) == 0:
        raise Exception(
            "No face detected in selfie"
        )

    faceid_embeds = torch.from_numpy(
        faces[0].normed_embedding
    ).unsqueeze(0).to(
        DEVICE,
        dtype=torch.float16
    )

    # ============================================
    # LOAD POSE IMAGE
    # ============================================

    pose_image = Image.open(
        pose_path
    ).convert("RGB")

    pose_image = pose_image.resize(
        (1024, 1792)
    )

    # ============================================
    # PROMPT
    # ============================================

    prompt = """
    ultra realistic full body human,
    preserve original pose,
    preserve original body shape,
    preserve original clothes,
    preserve original anatomy,
    preserve original proportions,
    realistic skin texture,
    realistic eyes,
    realistic hair,
    DSLR photography,
    cinematic lighting,
    photorealistic,
    highly detailed realistic person
    """

    negative_prompt = """
    cartoon,
    anime,
    3d render,
    cgi,
    blurry,
    deformed face,
    extra limbs,
    mutated body,
    ugly,
    fake skin,
    unrealistic anatomy,
    distorted eyes,
    malformed face,
    oversmoothed skin,
    duplicate body
    """

    # ============================================
    # GENERATE
    # ============================================

    images = ip_model.generate(
        face_image=face_image,
        image=pose_image,

        faceid_embeds=faceid_embeds,

        prompt=prompt,

        negative_prompt=negative_prompt,

        strength=0.32,

        num_inference_steps=40,

        guidance_scale=6.5,
    )
    image = images[0]

    # ============================================
    # SAVE
    # ============================================

    image.save(
        output_path
    )

    return image