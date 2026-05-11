import sys
from pathlib import Path

sys.path.append("/root/arc_3/IP-Adapter")

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insightface_model

from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL
from step_1.insightface_arcface import get_embedding_obj


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
OUTPUT_SIZE = (1024, 1792)

_PIPE = None
_FACE_ANALYZER = None
_IP_MODEL = None
_FACE_SWAPPER = None


def _get_providers():
    if DEVICE == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _ensure_models_loaded():
    global _PIPE, _FACE_ANALYZER, _IP_MODEL, _FACE_SWAPPER

    if _PIPE is None:
        _PIPE = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=DTYPE,
            variant="fp16" if DEVICE == "cuda" else None,
        ).to(DEVICE)

        if DEVICE == "cuda":
            try:
                _PIPE.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    if _FACE_ANALYZER is None:
        _FACE_ANALYZER = FaceAnalysis(
            name="buffalo_l",
            providers=_get_providers(),
        )
        _FACE_ANALYZER.prepare(
            ctx_id=0 if DEVICE == "cuda" else -1,
            det_size=(640, 640),
        )

    if _IP_MODEL is None:
        _IP_MODEL = IPAdapterFaceIDPlusXL(
            _PIPE,
            "/root/arc_3/models/image_encoder/IP-Adapter/models/image_encoder",
            "/root/arc_3/models/ip-adapter-faceid-plusv2_sdxl.bin",
            DEVICE,
        )

    if _FACE_SWAPPER is None:
        try:
            _FACE_SWAPPER = get_insightface_model("inswapper_128.onnx", providers=_get_providers())
        except Exception as exc:
            print(f"[WARN] Face swapper not loaded, skipping final face lock: {exc}")
            _FACE_SWAPPER = False

    return _PIPE, _FACE_ANALYZER, _IP_MODEL, _FACE_SWAPPER


def _select_primary_face(faces):
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def _expand_bbox(bbox, image_shape, expand_ratio=0.35):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * expand_ratio)
    pad_h = int(bh * expand_ratio)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w, x2 + pad_w)
    ny2 = min(h, y2 + pad_h)
    return nx1, ny1, nx2, ny2


def _resize_with_padding(pil_image, target_size):
    target_w, target_h = target_size
    src_w, src_h = pil_image.size
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(src_w * scale))
    resized_h = max(1, int(src_h * scale))

    resized = pil_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _apply_face_lock_swap(source_bgr, target_pil, face_analyzer, face_swapper):
    """
    Final identity lock: replace generated face region with source identity face
    using InsightFace swapper. This keeps placement from generated output while
    correcting distorted eyes/lips and identity drift.
    """
    if not face_swapper:
        return target_pil

    target_bgr = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)

    src_faces = face_analyzer.get(source_bgr)
    tgt_faces = face_analyzer.get(target_bgr)

    if not src_faces or not tgt_faces:
        return target_pil

    src_face = _select_primary_face(src_faces)
    tgt_face = _select_primary_face(tgt_faces)

    swapped_bgr = face_swapper.get(target_bgr, tgt_face, src_face, paste_back=True)
    swapped_rgb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(swapped_rgb)


@torch.inference_mode()
def generate_identity_transfer_stable(
    selfie_path: str,
    pose_path: str,
    output_path: str = "final_output.png",
):
    _, face_analyzer, ip_model, face_swapper = _ensure_models_loaded()

    selfie = cv2.imread(selfie_path)
    if selfie is None:
        print(f"[SKIP] Selfie not loaded: {selfie_path}")
        return None

    embedding_obj = get_embedding_obj(selfie.copy(), save_image=False)
    if embedding_obj is not None:
        bbox = embedding_obj["bbox"]
        embedding = embedding_obj["embedding"]
    else:
        # InsightFace expects BGR.
        faces = face_analyzer.get(selfie)
        if len(faces) == 0:
            print(f"[SKIP] No face detected in selfie: {selfie_path}")
            return None
        primary_face = _select_primary_face(faces)
        bbox = primary_face.bbox
        embedding = primary_face.embedding

    ex1, ey1, ex2, ey2 = _expand_bbox(bbox, selfie.shape, expand_ratio=0.35)
    face_crop = selfie[ey1:ey2, ex1:ex2]
    if face_crop.size == 0:
        print(f"[SKIP] Invalid face crop from selfie: {selfie_path}")
        return None

    face_crop = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    face_image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

    faceid_embeds = torch.from_numpy(np.array(embedding, dtype=np.float32)).unsqueeze(0).to(
        DEVICE,
        dtype=DTYPE,
    )

    try:
        pose_image = _resize_with_padding(Image.open(pose_path).convert("RGB"), OUTPUT_SIZE)
    except Exception as exc:
        print(f"[SKIP] Pose image invalid: {pose_path} ({exc})")
        return None

    prompt = """
    RAW photo, ultra realistic human portrait, preserve exact identity,
    same person as reference selfie, natural skin texture, realistic pores,
    realistic eyes, realistic hairline, realistic facial proportions,
    clear eyes, visible iris details, natural lips, natural teeth, neutral expression,
    preserve original pose, preserve original body shape, preserve original clothes,
    true-to-life color, high detail, 85mm lens, studio photo
    """

    negative_prompt = """
    cartoon, anime, painting, illustration, 3d render, cgi, doll, waxy skin,
    plastic skin, airbrushed face, over-processed skin, unrealistic face,
    deformed face, distorted eyes, malformed face, asymmetrical eyes,
    cross-eyed, lazy eye, broken mouth, warped smile, malformed lips, bad teeth,
    extra limbs, mutated body, blurry, lowres, duplicate body
    """

    try:
        images = ip_model.generate(
            face_image=face_image,
            image=pose_image,
            faceid_embeds=faceid_embeds,
            prompt=prompt,
            negative_prompt=negative_prompt,
            scale=1.35,
            s_scale=0.9,
            strength=0.18,
            num_samples=1,
            num_inference_steps=60,
            guidance_scale=4.2,
        )
        image = images[0]
        image = _apply_face_lock_swap(selfie, image, face_analyzer, face_swapper)
    except Exception as exc:
        print(f"[SKIP] Stable identity transfer failed for {selfie_path}: {exc}")
        return None

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_file))
    return image
