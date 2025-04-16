# register_faces.py - Register new faculty faces
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

DATA_PATH = 'data_faces'
os.makedirs(DATA_PATH, exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def register_face(name):
    cap = cv2.VideoCapture(0)
    embeddings = []
    print(f"Registering face for {name}...")
    count = 0

    while count < 10:
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)
        if face is not None:
            emb = resnet(face.unsqueeze(0)).detach().numpy()[0]
            embeddings.append(emb)
            count += 1
            cv2.putText(frame, f"Captured: {count}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Registering", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if embeddings:
        avg_emb = np.mean(embeddings, axis=0)
        np.save(os.path.join(DATA_PATH, f"{name}.npy"), avg_emb)
        print(f"Successfully registered {name}")
    else:
        print("Face not detected. Try again.")

if __name__ == "__main__":
    name = input("Enter name of faculty to register: ")
    register_face(name)
