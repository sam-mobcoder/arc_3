from step_3.stable_diffusion import generate_image

if __name__ == "__main__":
    img_path = "face/face_1.png"
    pose_path = "pose/pose_1.png"

    result = generate_image(
        selfie_path=img_path,
        pose_path=pose_path,
    )
    result.save("result.png")


