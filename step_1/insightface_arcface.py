'''
    Input :- Face Image
    Output :- {
        "embedding":[],
        "landmarks":[],
        "bbox":[]
    }
'''

from insightface.app import FaceAnalysis
import cv2
import onnxruntime as ort

app = FaceAnalysis(
    name="buffalo_l",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))


def get_embedding_obj(img, save_image=False):
    faces = app.get(img)
    if len(faces) == 0 or len(faces) > 1:
        return None

    bbox = faces[0].bbox
    landmarks = faces[0].kps
    # InstantID expects the ArcFace embedding vector (not the L2-normalized variant).
    embedding = faces[0].embedding
    gender = faces[0].gender
    age = faces[0].age

    # saving image with boundry box
    x1, y1, x2, y2 = map(int,bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    for (x, y) in landmarks:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 2)
    
    if save_image:
        cv2.imwrite("face/face_with_boundry_box.png", img)

        # Saving embedding into file
        with open("embedding/face_embedding.txt", "w") as f:
            f.write(str(embedding))


    return {
        "bbox": bbox,
        "landmarks": landmarks,
        "embedding": embedding,
        "gender": 'Female' if gender == 0 else "Male",
        "age": age
    }