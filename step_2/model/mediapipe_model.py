import mediapipe
import cv2
import numpy as np


mp_pose = mediapipe.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True
)

def get_body_features(image, bbox, landmarks):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    h, w, _ = image.shape

    # Face width
    x1, y1, x2, y2 = map(int, bbox)
    face_width = x2-x1
    result = pose.process(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    body_type = "average"
    shoulder_width = None
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        left_shoulder = lm[11]
        right_shoulder = lm[12]

        shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w

        # simple heurastic ratio
        ratio = shoulder_width / face_width 
        
        if ratio < 2.0:
            body_type = "slim"
        elif ratio > 2.5:
            body_type = "average"
        else:
            body_type = "broad"

    return {
        "body_type": body_type,
        "shoulder_width": shoulder_width,
        "face_ratio": face_width / w
    }