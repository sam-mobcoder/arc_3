# import cv2
# from step_1.instant_id_data_prepare import prepare_data
# from step_2.pose_estimation import get_pose_estimation
# from step_2.body_estimation import get_body_estimation
# from step_3.stable_diffusion_model import load_pipeline

# def generate_image(
#     selfie_path,
#     pose_path,
# ):
#     selfie_image = cv2.imread(selfie_path)

#     # Extract Identity - Step 1
#     identity = prepare_data(selfie_path)

    
#     # Extract Pose and body  - Step 2
#     pose_features = get_pose_estimation(pose_path)

#     body_estimation = get_body_estimation(
#         selfie_path, 
#         identity['bbox'], 
#         identity['landmarks']
#     )

#     print('body_estimation:-', body_estimation)

#     pipe = load_pipeline()

#     # setting instant id
#     pipe.set_instant_id(
#         embedding_tensor=identity['embedding_tensor'],
#         landmarks_tensor=identity['landmarks_tensor']
#     )

#     # prompt
#     prompt = f"""
#     Full body photo, {body_estimation['body_type']} bod, realistic, studio lighting
#     """

#     # negative prompt
#     negative_prompt = "bad anatomy, blurry, deformed face"

#     image = pipe(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         image=pose_features,
#         ip_adapter_embeds=identity['embedding_tensor'],
#         num_inference_steps=30,
#         guidance_scale=7.5
#     ).images[0]

#     return image




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
    face_kps = draw_kps(selfie_image, identity["landmarks"])

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
    prompt = f"""
    RAW photo, ultra realistic portrait of a {identity['gender']} person,
    age around {identity['age']},
    same identity, same facial structure, same eyes, same nose, same lips,
    {body_estimation['body_type']} body type,
    natural skin texture, pores, realistic hair strands,
    professional photography, soft cinematic lighting,
    85mm lens, shallow depth of field, high dynamic range,
    highly detailed, photorealistic
    """

    negative_prompt = """
    cartoon, anime, painting, 3d render, cgi, doll, plastic skin, waxy skin,
    blurry, out of focus, low quality, low resolution, jpeg artifacts,
    bad anatomy, deformed face, asymmetrical eyes, distorted mouth,
    extra limbs, duplicate face, mutated hands, disfigured body
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
        image=[face_kps, pose_map["pose_map"]],

        # generation params
        width=width,
        height=height,
        ip_adapter_scale=1.15,
        controlnet_conditioning_scale=[1.0, 0.35],
        num_inference_steps=50,
        guidance_scale=4.5,
        generator=generator,
    ).images[0]

    return image