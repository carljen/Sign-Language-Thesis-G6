import cv2
import numpy as np
import os
import mediapipe as mp

# ==========================================
# 1. SETUP PATHS & ACTIONS
# ==========================================
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['b','e'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Create the folder structure automatically
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# ==========================================
# 2. SETUP MEDIAPIPE
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==========================================
# 3. HELPER FUNCTION (Extract Keypoints)
# ==========================================
def extract_keypoints(results):
    # Pose (33*4 = 132)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Left Hand (21*3 = 63)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # Right Hand (21*3 = 63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatenate (Total 258 points) -> NO FACE MESH
    return np.concatenate([pose, lh, rh])

# ==========================================
# 4. MAIN DATA COLLECTION LOOP
# ==========================================
cap = cv2.VideoCapture(0)

# Set model complexity to 1 to match your real-time speed
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as holistic:
    
    # Loop through actions (hello -> thanks -> iloveyou)
    for action in actions:
        # Loop through sequences (videos) 0 to 29
        for sequence in range(no_sequences):
            # Loop through video length (frames) 0 to 29
            for frame_num in range(sequence_length):

                # Read Feed
                ret, frame = cap.read()

                # Make detections
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                # Draw landmarks
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw Pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                # Draw Hands
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # NEW: Apply wait logic
                if frame_num == 0: 
                    # Display "STARTING COLLECTION" for 2 seconds
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000) # Wait 2 seconds
                else: 
                    # Just show the collection status
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                # EXPORT KEYPOINTS
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()