'''
INPUT :- generated scene image

OUTPUT :-{
    "face_mask": mask,
    "face_bbox": bbox,
}
'''

import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis


def _largest_face(faces):
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


class FaceMaskGenerator:

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self._insight_app = None

    def _get_insight(self):
        if self._insight_app is None:
            self._insight_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._insight_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._insight_app

    def _mask_from_insightface(self, image_rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self._get_insight().get(bgr)
        if not faces:
            raise RuntimeError("No face detected (InsightFace fallback)")
        bbox = _largest_face(faces).bbox
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image_rgb.shape[:2]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rw = int((x2 - x1) * 0.65)
        rh = int((y2 - y1) * 0.85)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (max(rw, 1), max(rh, 1)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        return mask

    def generate_mask(self, image):

        # MediaPipe FaceMesh expects RGB input.
        image_rgb = np.array(image)

        h, w, _ = image_rgb.shape

        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            points = []

            for lm in landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append([x, y])

            points = np.array(points)

            hull = cv2.convexHull(points)

            mask = np.zeros((h, w), dtype=np.uint8)

            cv2.fillConvexPoly(mask, hull, 255)

            mask = cv2.GaussianBlur(mask, (31, 31), 0)

            return mask

        try:
            return self._mask_from_insightface(image_rgb)
        except Exception as exc:
            raise RuntimeError(
                f"No face detected (MediaPipe + InsightFace): {exc}"
            ) from exc