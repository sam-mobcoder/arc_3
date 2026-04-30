import cv2
from .step_1.instant_id_data_prepare import prepare_data
from .step_2.pose_estimation import get_pose_estimation
from .step_2.body_estimation import get_body_estimation
from .stable_diffusion_model import load_pipeline

def generate_image(
    selfie_path,
    pose_path,
):
    selfie_image = cv2.imread(selfie_path)

    # Extract Identity - Step 1
    identity = prepare_data(selfie_path)

    # Extract Pose and body  - Step 2
    pose_features = get_pose_estimation(pose_path)
    body_estimation = get_body_estimation(
        selfie_path, 
        identity['bbox'], 
        identity['landmarks']
    )

    pipe = load_pipeline()

    # setting instant id
    pipe.set_instand_id(
        embedding_tensor=identity['embedding_tensor'],
        landmarks_tensor=identity['landmarks_tensor']
    )

    # prompt
    prompt = f"""
    Full body photo, {body_estimation['body_type']} bod, realistic, studio lighting
    """

    # negative prompt
    negative_prompt = "bad anatomy, blurry, deformed face"

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pose_features,
        ip_adapter_embeds=identity['embedding_tensor'],
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    return image