import os
import socket
import shutil
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
POSE_DIR = PROJECT_ROOT / "images" / "pose"
APP_OUTPUT_DIR = PROJECT_ROOT / "images" / "face_result" / "gradio_stable_latest"


def _list_pose_images():
    return sorted(
        [p for p in POSE_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    )


def _prepare_output_dir():
    if APP_OUTPUT_DIR.exists():
        shutil.rmtree(APP_OUTPUT_DIR)
    APP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_available_port(start_port: int, max_tries: int = 50) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free port found in range {start_port}-{start_port + max_tries - 1}. "
        "Set GRADIO_SD_SERVER_PORT to another value and retry."
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def generate_for_all_poses_stable(selfie_image):
    if selfie_image is None:
        raise gr.Error("Please upload a selfie image.")

    pose_images = _list_pose_images()
    if not pose_images:
        raise gr.Error(f"No pose images found in: {POSE_DIR}")

    # Lazy import keeps server startup fast.
    from step_3.stable_diffusion_pose_transfer import generate_identity_transfer_stable

    _prepare_output_dir()

    gallery_items = []
    total_poses = len(pose_images)
    yield gallery_items, f"Generating (Stable Diffusion)... 0/{total_poses} completed"

    for idx, pose_path in enumerate(pose_images, start=1):
        output_path = APP_OUTPUT_DIR / f"result_pose_{idx}{pose_path.suffix.lower()}"
        status_prefix = f"Generating pose {idx}/{total_poses}: {pose_path.name}"
        yield gallery_items, status_prefix

        result_image = generate_identity_transfer_stable(
            selfie_path=str(selfie_image),
            pose_path=str(pose_path),
            output_path=str(output_path),
        )

        if result_image is None:
            print(f"[SKIP] Stable generation failed for pose: {pose_path.name}")
            yield gallery_items, f"{status_prefix} (skipped)"
            continue

        gallery_items.append((str(output_path), f"Pose {idx}: {pose_path.name}"))
        yield gallery_items, f"Generating (Stable Diffusion)... {len(gallery_items)}/{total_poses} completed"

    yield gallery_items, f"Done: generated {len(gallery_items)}/{total_poses} outputs."


APP_CSS = """
#generated_gallery_sd, #generated_gallery_sd .grid-wrap, #generated_gallery_sd .wrap {
    max-height: none !important;
    overflow-y: visible !important;
}
"""


with gr.Blocks(title="Selfie Pose Transfer (Stable Diffusion)") as demo:
    gr.Markdown("## Selfie -> Multi Pose (Stable Diffusion Flow)")
    gr.Markdown(
        "Upload one selfie. The app applies SD-based identity transfer to all poses in `images/pose` and shows results progressively."
    )

    with gr.Row():
        selfie_input = gr.Image(
            type="filepath",
            label="Upload Selfie",
            elem_id="selfie_upload_sd",
            width=240,
            height=240,
        )
        output_gallery = gr.Gallery(
            label="Generated Outputs",
            elem_id="generated_gallery_sd",
            columns=2,
            height="auto",
        )

    status_text = gr.Textbox(label="Status", interactive=False)
    run_button = gr.Button("Generate All Poses (Stable)", variant="primary")

    run_button.click(
        fn=generate_for_all_poses_stable,
        inputs=[selfie_input],
        outputs=[output_gallery, status_text],
    )


if __name__ == "__main__":
    preferred_port = int(os.getenv("GRADIO_SD_SERVER_PORT", "7863"))
    launch_port = _find_available_port(preferred_port, max_tries=50)
    if launch_port != preferred_port:
        print(f"[INFO] Port {preferred_port} is busy. Launching Gradio on {launch_port} instead.")
    use_share = _env_flag("GRADIO_SHARE", default=True)

    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=launch_port,
        share=use_share,
        show_error=True,
        css=APP_CSS,
    )
