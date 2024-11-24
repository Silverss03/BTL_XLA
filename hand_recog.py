import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the saved CNN model
model_path = r"final.h5"
model = load_model(model_path)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Update to match training classes
alphabet_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
                  'W', 'X', 'Y']

def preprocess_image(img):
    # Resize to match training input size
    img = cv2.resize(img, (200, 200))
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def extract_hand_region(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get bounding box coordinates
        h, w, _ = frame.shape
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        # Increase padding to 60 pixels (was 30)
        padding = 60
        x_min, x_max = int(max(0, min(x_coords) - padding)), int(min(w, max(x_coords) + padding))
        y_min, y_max = int(max(0, min(y_coords) - padding)), int(min(h, max(y_coords) + padding))
        
        # Add extra vertical padding at the bottom for gestures that extend downward
        y_max = min(h, y_max + 20)
        
        # Crop hand region
        hand_region = frame[y_min:y_max, x_min:x_max]
        
        return hand_region, (x_min, y_min, x_max, y_max)
    return None, None

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract hand region
    hand_crop, bbox = extract_hand_region(frame)
    
    if hand_crop is not None and hand_crop.size > 0:
        # Process the cropped hand region
        preprocessed_hand = preprocess_image(hand_crop)
        predictions = model.predict(preprocessed_hand)
        predicted_class = alphabet_labels[np.argmax(predictions)]

        # Draw bounding box and prediction
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Predicted: {predicted_class}",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

    cv2.imshow("ASL Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()