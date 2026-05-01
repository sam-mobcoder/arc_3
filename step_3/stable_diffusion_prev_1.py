import cv2
import torch
from PIL import Image

from step_1.instant_id_data_prepare import prepare_data
from step_2.pose_estimation import get_pose_estimation
from step_2.body_estimation import get_body_estimation
from step_3.stable_diffusion_model import load_pipeline
from InstantID.pipeline_stable_diffusion_xl_instantid import draw_kps


def generate_image(
    selfie_path,
    pose_path,
):
    def _target_resolution_from_pose(pose_image, max_side=1024, min_side=768):
        width, height = pose_image.size
        long_side = max(width, height)
        scale = max_side / float(long_side)
        scaled_w = int(round(width * scale))
        scaled_h = int(round(height * scale))

        # Keep SDXL-compatible dimensions.
        scaled_w = max(min_side, (scaled_w // 64) * 64)
        scaled_h = max(min_side, (scaled_h // 64) * 64)
        return scaled_w, scaled_h

    # -----------------------------
    # STEP 1 — Identity
    # -----------------------------
    identity = prepare_data(selfie_path)

    """
    identity = {
        "embedding_tensor": ...,
        "landmarks_tensor": ...,
        "bbox": ...,
        "landmarks": ...
    }
    """

    # -----------------------------
    # STEP 2A — Pose
    # -----------------------------
    pose_map = get_pose_estimation(pose_path)
    selfie_image = Image.open(selfie_path).convert("RGB")
    # face_kps = draw_kps(selfie_image, identity["landmarks"])
    face_kps = identity["face_pil"]

    # -----------------------------
    # STEP 2B — Body
    # -----------------------------
    body_estimation = get_body_estimation(
        selfie_path,
        identity["bbox"],
        identity["landmarks"]
    )

    print("body_estimation:-", body_estimation)

    # -----------------------------
    # STEP 3 — Diffusion Pipeline
    # -----------------------------
    pipe = load_pipeline()

    # prompt
    # prompt = f"""
    # a person {identity['gender']} person,
    # age around {identity['age']},
    # same identity, same facial structure, same eyes, same nose, same lips,
    # {body_estimation['body_type']} body type,
    # natural skin texture, pores, realistic hair strands,
    # professional photography, soft cinematic lighting,
    # 85mm lens, shallow depth of field, high dynamic range,
    # highly detailed, photorealistic
    # """

    prompt = f"""
    photorealistic full body human
    """

    negative_prompt = """
    low quality, blurry, distorted face, deformed anatomy
    """

    # -----------------------------
    # GENERATION
    # -----------------------------
    width, height = _target_resolution_from_pose(pose_map["pose_map"])
    generator = torch.Generator(device="cuda").manual_seed(123456)

    image = pipe(
        prompt=prompt,

        negative_prompt=negative_prompt,

        # InstantID embedding
        image_embeds=identity["embedding_tensor"],

        # Multi-ControlNet inputs: [InstantID face control, OpenPose body control]
        # image=[face_kps, pose_map["pose_map"]],
        image=face_kps,

        # generation params
        width=width,
        height=height,
        ip_adapter_scale=0.8,
        # controlnet_conditioning_scale=[1.0, 0.35],
        controlnet_conditioning_scale=1.0,
        num_inference_steps=50,
        guidance_scale=5,
        generator=generator,
    ).images[0]

    return image