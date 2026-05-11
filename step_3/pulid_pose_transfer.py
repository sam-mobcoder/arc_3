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
OUTPUT_SIZE = (1024, 1792)

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

    # Prefer the largest detected face for stable identity extraction.
    primary_face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    faceid_embeds = torch.from_numpy(
        primary_face.normed_embedding
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

    pose_image = pose_image.resize(OUTPUT_SIZE)

    # ============================================
    # PROMPT
    # ============================================

    prompt = """
    RAW photo, ultra realistic human portrait, preserve exact identity,
    same person as reference selfie, natural skin texture, realistic pores,
    realistic eyes, realistic hairline, realistic facial proportions,
    preserve original pose, preserve original body shape, preserve original clothes,
    true-to-life color, high detail, 85mm lens, studio photo
    """

    negative_prompt = """
    cartoon, anime, painting, illustration, 3d render, cgi, doll, waxy skin,
    plastic skin, airbrushed face, over-processed skin, unrealistic face,
    deformed face, distorted eyes, malformed face, asymmetrical eyes,
    extra limbs, mutated body, blurry, lowres, duplicate body
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

        scale=1.2,
        s_scale=1.1,
        strength=0.25,

        num_samples=1,
        num_inference_steps=55,

        guidance_scale=5.0,
    )
    image = images[0]

    # ============================================
    # SAVE
    # ============================================

    image.save(
        output_path
    )

    return image