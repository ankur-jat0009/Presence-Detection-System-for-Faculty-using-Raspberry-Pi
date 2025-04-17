# Cam1.py - Entrance Recognition

import cv2, os, json, numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Paths to store known face embeddings and faculty status
DATA_PATH = 'data_faces'
STATUS_PATH = 'status.json'

# Initialize face detector (MTCNN) and face recognizer (InceptionResnetV1)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load known face embeddings from saved .npy files
known_embeddings = {
    f[:-4]: np.load(os.path.join(DATA_PATH, f))
    for f in os.listdir(DATA_PATH) if f.endswith('.npy')
}

# Function to recognize a face from the video frame
def recognize_face(face_img):
    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))  # Convert to RGB and PIL format
    face = mtcnn(img)  # Detect face
    if face is None:
        return None
    emb = resnet(face.unsqueeze(0)).detach().numpy()[0]  # Generate embedding
    # Compare with stored embeddings
    for name, db_emb in known_embeddings.items():
        dist = np.linalg.norm(emb - db_emb)
        if dist < 0.9:  # Threshold for face recognition
            return name
    return None

# Start video capture (camera at entrance)
cap = cv2.VideoCapture(0)
print("[Cam1] Entrance camera started...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    name = recognize_face(frame)  # Try to recognize face from frame

    if name:
        # Update status.json to mark the faculty as "present"
        with open(STATUS_PATH, 'r+') as f:
            data = json.load(f)
            data[name] = "present"
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        # Display recognition result on frame
        cv2.putText(frame, f"{name} entered", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Entrance - Cam1", frame)  # Show video frame

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
