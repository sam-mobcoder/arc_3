from .insightface_arcface import get_embedding_obj

import cv2
import torch

from PIL import Image


def prepare_pulid_data(img_path):

    img = cv2.imread(img_path)

    embedding_obj = get_embedding_obj(
        img,
        save_image=False
    )

    if embedding_obj is None:
        raise Exception(
            "Invalid image. Only one face should be present."
        )

    # -----------------------------------
    # FACE CROP
    # -----------------------------------
    x1, y1, x2, y2 = map(int, embedding_obj["bbox"])

    face_crop = img[y1:y2, x1:x2]

    # IMPORTANT:
    # larger crop preserves hairstyle/jawline
    pad = 140

    h, w, _ = img.shape

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)

    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    face_crop = img[y1:y2, x1:x2]

    # FLUX/PuLID likes higher quality reference
    face_crop = cv2.resize(face_crop, (768, 768))

    # BGR -> RGB
    face_crop_rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    face_pil = Image.fromarray(face_crop_rgb)

    # -----------------------------------
    # EMBEDDING
    # -----------------------------------
    embedding_tensor = (
        torch.tensor(embedding_obj["embedding"])
        .unsqueeze(0)
        .float()
        .cuda()
    )

    return {
        "face_pil": face_pil,
        "embedding_tensor": embedding_tensor,
        "gender": embedding_obj["gender"],
        "age": embedding_obj["age"],
        "bbox": (x1, y1, x2, y2)
    }