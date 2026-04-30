# this file will prepare data for the Instant ID Model which will run at step 3 with diffusion model

from .insightface_arcface import get_embedding_obj
import cv2
import torch
from PIL import Image

def prepare_data(img_path):
    img = cv2.imread(img_path)
    embedding_obj = get_embedding_obj(
        img, 
        save_image=False
    )

    if embedding_obj is None:
        print("Invalid Image.. Make sure that only one face is present in the image")
        exit()

    # preparing face crop input for Instant ID
    x1, y1, x2, y2  = map(int, embedding_obj['bbox'])
    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (224, 224))


    # convert into tensors
    embedding_tensor = torch.tensor(embedding_obj['embedding']).unsqueeze(0).float().cuda()
    landmarks_tensor = torch.tensor(embedding_obj['landmarks']).unsqueeze(0).float().cuda()

    # face_pil
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

    return {
        "face_pil": face_pil,
        "embedding_tensor": embedding_tensor,
        "landmarks_tensor": landmarks_tensor,
        "embedding": embedding_obj['embedding'],
        "landmarks": embedding_obj['landmarks'],
        "bbox": (x1, y1, x2, y2),
        "gender": embedding_obj['gender'],
        "age": embedding_obj['age']
    }
