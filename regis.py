# Import required libraries
import cv2  # For accessing the webcam and displaying frames
import os  # For file operations
import numpy as np  # For handling numerical operations (like averaging embeddings)
from facenet_pytorch import MTCNN, InceptionResnetV1  # Face detection and embedding models
from PIL import Image  # To convert OpenCV images for MTCNN
import torch  # Required for PyTorch models
import time  # For timeout mechanism

# Directory to save registered face embeddings
DATA_PATH = 'data_faces'
os.makedirs(DATA_PATH, exist_ok=True)

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device='cpu')

# Load pre-trained InceptionResnetV1 for face embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to register a new faculty face
def register_face(name):
    cap = cv2.VideoCapture(0)  # Start webcam
    embeddings = []  # To store multiple face embeddings
    print(f"Registering face for {name}...")
    count = 0  # Counter for how many embeddings we collect
    timeout = time.time() + 60  # Limit registration time to 1 minute

    while count < 10 and time.time() < timeout:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert frame to RGB and then to PIL Image for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Detect faces in the image
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            for box in boxes:
                # Draw rectangle around detected face
                cv2.rectangle(frame, 
                              (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), 
                              (0, 255, 0), 2)

            # Extract aligned face tensor
            face_tensor = mtcnn(img)
            if face_tensor is not None:
                # Generate face embedding
                emb = resnet(face_tensor.unsqueeze(0)).detach().numpy()[0]
                embeddings.append(emb)
                count += 1
                print(f"Captured {count}/10")

        # Display capture progress on screen
        cv2.putText(frame, f"Captured: {count}/10", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Registering", frame)

        # Allow early exit using 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # If embeddings collected, average and save
    if embeddings:
        avg_emb = np.mean(embeddings, axis=0)
        np.save(os.path.join(DATA_PATH, f"{name}.npy"), avg_emb)
        print(f"Successfully registered {name}")
    else:
        print("Face not detected. Try again.")

# Entry point to run the script directly
if __name__ == "__main__":
    name = input("Enter name of faculty to register: ")
    register_face(name)
