
import sys, os
sys.path.append("/root/arc_3/InstantID")
from step_3.stable_diffusion import generate_image
from tqdm import tqdm  

if __name__ == "__main__":

    num_inference_steps = [35, 45, 55]
    guidance_scales = [2.5, 3.0, 3.5]
    id_weights = [0.7, 0.9, 1.1]

    for image_path in tqdm(os.listdir("images/face")):
        for num_inference_step in num_inference_steps:
            for guidance_scale in guidance_scales:
                for id_weight in id_weights:
                    result = generate_image(
                        selfie_path=f"images/face/{image_path}",
                        # pose_path=pose_path,
                        num_inference_steps=num_inference_step,
                        guidance_scale=guidance_scale,
                        id_weight=id_weight,
                    )

                    if result is None:
                        print(f"[SKIP] Generation skipped for: {image_path}")
                        continue

                    result.save(f"images/face_result/{image_path.split('.')[0]}_{guidance_scale}_{id_weight}_{num_inference_step}.png", format="png")
                    print(f"Saved generated image to: {image_path.split('.')[0]}_{guidance_scale}_{id_weight}_{num_inference_step}.png")


