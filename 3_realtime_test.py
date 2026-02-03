import os
# 1. Hide messy TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 2. IMPORT TENSORFLOW FIRST (Prevents the freeze)
import tensorflow as tf
from tensorflow.keras.models import load_model

# 3. Import everything else afterwards
import cv2 
import numpy as np
import mediapipe as mp
import time

# Load your trained model
model = load_model('action.h5') # Make sure your file is named 'action.h5'

# Define your actions (Must match the order you trained on!)
# Example: actions = np.array(['hello', 'thanks', 'iloveyou']) 
actions = np.array(['hello', 'thanks', 'iloveyou', 'a'  , 'b'])  # Updated to include 'a' and 'b'
# ==========================================
# CONFIGURATION AREA
# ==========================================
OUTPUT_FPS = 15.0 
MODEL_COMPLEXITY = 1 

# ==========================================
# HELPER FUNCTION: EXTRACT KEYPOINTS
# ==========================================
def extract_keypoints(results):
    # 1. Pose (33 points * 4 dims = 132)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # 2. REMOVED FACE EXTRACTION HERE (This was adding ~1400 extra points)
    
    # 3. Left Hand (21 points * 3 dims = 63)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # 4. Right Hand (21 points * 3 dims = 63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # 5. Concatenate ONLY Pose and Hands
    return np.concatenate([pose, lh, rh])

# ==========================================
# SETUP MEDIAPIPE & CAMERA
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_collection.avi', fourcc, OUTPUT_FPS, (width, height))

prev_time = 0
frame_num = 0
is_recording_data = False # Toggle to control data collection

# ==========================================
# MAIN LOOP
# ==========================================
with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=MODEL_COMPLEXITY,
    static_image_mode=False
) as holistic:
    
    print("Press 'r' to toggle data collection ON/OFF")
    print("Press 'q' to quit")
    sequence = []
    sentence = []
    threshold = 0.8 # Confidence threshold (only show result if 80% sure)

    while cap.isOpened():
        
        success, image = cap.read()
        if not success:
            continue
        
        start_time = time.time()
        # 1. Detection
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # 2. Draw Landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # ==========================================
        # PREDICTION LOGIC
        # ==========================================
        # 1. Extract Keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep only the last 30 frames

        # 2. Predict (Only once we have 30 frames)
        # 2. Predict (Only once we have 30 frames)
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            
            # Get the highest confidence action
            best_idx = np.argmax(res)
            current_action = actions[best_idx]
            confidence = res[best_idx]

            # --- DEBUGGING: Print to Console ---
            # This helps you see if the model works at all. 
            # If you see numbers changing in your terminal, the AI is working!
            print(f"Action: {current_action} | Confidence: {confidence:.2f}")

            # Logic to build the sentence (Only if confident)
            if confidence > threshold: 
                if len(sentence) > 0: 
                    if current_action != sentence[-1]:
                        sentence.append(current_action)
                else:
                    sentence.append(current_action)

            # Limit sentence length
            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # --- VISUALIZATION ---
            # 1. Draw the "Locked In" Sentence (Blue Bar at Top)
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 2. Draw the "Live Status" (What it sees RIGHT NOW) - NEW!
            # This shows you what it is thinking, even if it's not confident enough to lock it in.
            status_text = f"Live: {current_action} ({int(confidence*100)}%)"
            
            # Color: Green if confident, Red if unsure
            text_color = (0, 255, 0) if confidence > threshold else (0, 0, 255)
            
            cv2.putText(image, status_text, (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

        # ==========================================
        # FPS & VIDEO RECORDING
        # ==========================================
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 2. STOP THE STOPWATCH (Place it here!)
        # =================================================
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Print it to the black terminal so you can see the speed
        print(f"Latency: {latency_ms:.2f} ms")
        # =================================================
        # Save video frame
        out.write(image)
        cv2.imshow('Data Collection Feed', image)

        # Controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()