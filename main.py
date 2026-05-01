
import sys
sys.path.append("/root/arc_3/InstantID")
from step_3.stable_diffusion import generate_image


if __name__ == "__main__":
    img_path = "face/face.png"
    pose_path = "pose/pose_1.png"
    output_path = "result.png"

    result = generate_image(
        selfie_path=img_path,
        pose_path=pose_path,
    )

    result.save(output_path)
    print(f"Saved generated image to: {output_path}")


