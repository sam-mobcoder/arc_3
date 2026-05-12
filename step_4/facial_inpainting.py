'''
INPUT :- {
    "target_image": scene_image,
    "face_embedding": face_embedding,
    "mask": face_mask,
    "prompt": prompt,
}

STAGE A :- Encode image into latent
STAGE B :- Apply noise Only masked face image
STAGE C :- Inject Identity Embedding
STAGE D:- Denoise ONLY Masked Region.

OUTPUT :- {
    "inpainted_image": inpainted_image,
}
'''
import torch
import numpy as np
import cv2

from einops import rearrange
from PIL import Image

from step_4.face_mask import FaceMaskGenerator
from step_4.latent_masking import LatentMasking


class FacialInpainting:

    def __init__(
        self,
        model,
        ae,
        t5,
        clip,
        pulid_model,
        device,
        dtype,
        prepare,
        denoise,
        unpack,
        get_schedule,
    ):
        self.model = model
        self.ae = ae
        self.t5 = t5
        self.clip = clip
        self.pulid_model = pulid_model

        self.device = device
        self.dtype = dtype

        self.prepare = prepare
        self.denoise = denoise
        self.unpack = unpack
        self.get_schedule = get_schedule

        self.mask_generator = FaceMaskGenerator()

        self.latent_masking = LatentMasking(
            ae=self.ae,
            device=self.device,
        )

    @torch.inference_mode()
    def regenerate_face(
        self,
        target_image,
        id_embeddings,
        prompt,
        reference_portrait=None,
        uncond_id=None,
        num_inference_steps=34,
        guidance_scale=3.8,
        id_weight=0.92,
    ):
        width, height = target_image.size

        # -----------------------------
        # FACE MASK
        # -----------------------------

        target_np = np.array(target_image).astype(np.float32)
        mask = self.mask_generator.generate_mask(
            target_image
        )

        # Tighter core: full denoised latent only in center; rim stays closer to original (less noise).
        kernel = np.ones((7, 7), np.uint8)
        mask_core = cv2.erode(mask, kernel, iterations=2)
        if int(mask_core.max()) == 0:
            mask_core = mask

        # -----------------------------
        # ENCODE IMAGE
        # -----------------------------

        latent = self.latent_masking.encode_image(
            target_image
        )
        latent_orig = latent.clone()

        # -----------------------------
        # APPLY MASKED NOISE
        # -----------------------------

        latent = self.latent_masking.apply_mask_noise(
            latent,
            mask,
        )

        # Keep latent/id embedding dtype aligned with the Flux/PuLID runtime dtype
        # (bfloat16 on CUDA, float32 on CPU) to avoid matmul dtype mismatch.
        latent = latent.to(self.device, dtype=self.dtype)
        id_embeddings = id_embeddings.to(self.device, dtype=self.dtype)
        if uncond_id is not None:
            uncond_id = uncond_id.to(self.device, dtype=self.dtype)

        # -----------------------------
        # TIMESTEPS
        # -----------------------------

        timesteps = self.get_schedule(
            num_inference_steps,
            latent.shape[-1] * latent.shape[-2] // 4,
            shift=True,
        )

        # -----------------------------
        # PREPARE PROMPT
        # -----------------------------

        inp = self.prepare(
            t5=self.t5,
            clip=self.clip,
            img=latent,
            prompt=prompt,
        )

        # -----------------------------
        # DENOISE
        # -----------------------------
        print("latent dtype:", latent.dtype)
        print("embedding dtype:", id_embeddings.dtype)

        x = self.denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=guidance_scale,
            id=id_embeddings,
            id_weight=id_weight,
            start_step=0,
            uncond_id=uncond_id,
            true_cfg=1.0,
        )

        # -----------------------------
        # DECODE
        # -----------------------------

        x = self.unpack(
            x.float(),
            height,
            width,
        )

        # PuLID denoise updates the full latent; blend back non-face (and face rim) from encode.
        mask_lat = cv2.resize(
            mask_core,
            (x.shape[-1], x.shape[-2]),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_t = torch.tensor(mask_lat, device=x.device, dtype=x.dtype) / 255.0
        mask_t = mask_t.view(1, 1, x.shape[-2], x.shape[-1])
        latent_orig_cast = latent_orig.to(device=x.device, dtype=x.dtype)

        if x.shape == latent_orig_cast.shape:
            x = x * mask_t + latent_orig_cast * (1.0 - mask_t)
        else:
            print(
                f"[WARN] Step4 latent shape mismatch decode={x.shape} orig={latent_orig_cast.shape}; "
                "skipping latent blend (may look noisy)."
            )

        with torch.autocast(
            device_type="cuda",
            enabled=False
        ):
            x = self.ae.decode(x.float())

        x = x.clamp(-1, 1)

        x = rearrange(
            x[0],
            "c h w -> h w c"
        )

        img = Image.fromarray(
            (
                127.5 * (x + 1.0)
            ).cpu().byte().numpy()
        )

        # Soft pixel blend at full face mask (smooth edge into pose body).
        regenerated_np = np.array(img).astype(np.float32)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        blended_np = regenerated_np * alpha + target_np * (1.0 - alpha)

        # Stabilize face core with Flux base portrait (same identity, no extra noise).
        if reference_portrait is not None:
            ref = reference_portrait.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            ref_np = np.array(ref).astype(np.float32)
            mcore = (mask_core.astype(np.float32) / 255.0)[..., None]
            stabilize = 0.28
            blended_np = blended_np * (1.0 - mcore * stabilize) + ref_np * (mcore * stabilize)

        blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)

        return Image.fromarray(blended_np)
