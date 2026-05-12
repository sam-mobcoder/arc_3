'''
INPUT :- scene_image, face_mask
OUTPUT :- masked_latent
'''

import torch
import numpy as np
import cv2


class LatentMasking:

    def __init__(self, ae, device):
        self.ae = ae
        self.device = device

    @torch.inference_mode()
    def encode_image(self, image):

        image = np.array(image).astype(np.float32)

        image = image / 127.5 - 1.0

        image = torch.tensor(image,dtype=torch.float32)

        image = image.permute(2, 0, 1).unsqueeze(0)

        image = image.to(self.device)

        with torch.autocast(
            device_type="cuda",
            enabled=False
        ):
            latent = self.ae.encode(image)

        return latent

    def apply_mask_noise(
        self,
        latent,
        mask,
        noise_strength=0.14,
    ):

        mask = cv2.resize(
            mask,
            (latent.shape[-1], latent.shape[-2])
        )

        mask = torch.tensor(mask, dtype=latent.dtype).to(self.device)

        mask = mask / 255.0

        mask = mask.unsqueeze(0).unsqueeze(0)

        noise = torch.randn_like(
            latent,
            dtype=latent.dtype
        )

        latent = latent * (1 - mask)

        latent = latent + (
            noise * mask * noise_strength
        )

        return latent