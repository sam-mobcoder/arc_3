# this file will prepare data for the IP Adapter Face ID Plus V2 Model which will run at step 3 with diffusion model
from PIL import Image
import cv2
import torch

def prepare_data(face_crop, embedding):
    face_pil = Image.fromarray(
        cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    )

    embedding_tensor = torch.tensor(embedding).unsqueeze(0).float().cuda()

    return {
        "ip_image":face_pil,
        "ip_embedding": embedding_tensor
    }