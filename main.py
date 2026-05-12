
import sys, os
sys.path.append("/root/arc_3/InstantID")
from step_3.stable_diffusion import generate_image
from tqdm import tqdm  

if __name__ == "__main__":

    guidance_scales = [3.0]
    id_weights = [1.35]
    num_inference_steps = [45]
    seeds = [2]

    for image_path in tqdm(os.listdir("images/face")):
        if image_path.split('.')[0] not in ['Selfie7677']:
            continue
        for seed in seeds:
            for num_inference_step in num_inference_steps:
                for guidance_scale in guidance_scales:
                    for id_weight in id_weights:
                        result = generate_image(
                            selfie_path=f"images/face/{image_path}",
                            pose_path=f"images/pose/pose_6.png",
                            num_inference_steps=num_inference_step,
                            guidance_scale=guidance_scale,
                            id_weight=id_weight,
                            seed=seed,
                        )

                        if result is None:
                            print(f"[SKIP] Generation skipped for: {image_path}")
                            continue

                        os.makedirs(f"images/face_result/{image_path.split('.')[0]}", exist_ok=True)
                        result.save(f"images/face_result/{image_path.split('.')[0]}/{seed}.png", format="png")
                        print(f"Saved generated image to: {image_path.split('.')[0]}/{seed}.png")


# from step_3.pulid_pose_transfer import (
#     generate_identity_transfer
# )

# generate_identity_transfer(
#     selfie_path="images/face/Selfie7677.jpg",
#     pose_path="images/pose/pose_1.png",
#     output_path="final_output.png"
# )