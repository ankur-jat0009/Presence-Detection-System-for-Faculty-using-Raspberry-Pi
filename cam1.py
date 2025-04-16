# Cam1.py - Entrance Recognition
import cv2, os, json, numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

DATA_PATH = 'data_faces'
STATUS_PATH = 'status.json'

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

known_embeddings = {f[:-4]: np.load(os.path.join(DATA_PATH, f)) for f in os.listdir(DATA_PATH) if f.endswith('.npy')}

def recognize_face(face_img):
    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    if face is None: return None
    emb = resnet(face.unsqueeze(0)).detach().numpy()[0]
    for name, db_emb in known_embeddings.items():
        dist = np.linalg.norm(emb - db_emb)
        if dist < 0.9: return name
    return None

cap = cv2.VideoCapture(0)
print("[Cam1] Entrance camera started...")

while True:
    ret, frame = cap.read()
    if not ret: continue
    name = recognize_face(frame)
    if name:
        with open(STATUS_PATH, 'r+') as f:
            data = json.load(f)
            data[name] = "present"
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        cv2.putText(frame, f"{name} entered", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Entrance - Cam1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
