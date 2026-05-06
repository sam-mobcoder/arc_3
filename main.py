
import sys, os
sys.path.append("/root/arc_3/InstantID")
from step_3.stable_diffusion import generate_image
from tqdm import tqdm  

if __name__ == "__main__":

    guidance_scales = [3.0]
    id_weights = [1.0]
    num_inference_steps = [50]
    seeds = [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    for image_path in tqdm(os.listdir("images/face")):
        for seed in seeds:
            for num_inference_step in num_inference_steps:
                for guidance_scale in guidance_scales:
                    for id_weight in id_weights:
                        result = generate_image(
                            selfie_path=f"images/face/{image_path}",
                            # pose_path=pose_path,
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


