import cv2
import numpy as np
import os
import mediapipe as mp

# ==========================================
# 1. SETUP PATHS & ACTIONS
# ==========================================
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(["Nice to meet you!"])

# Thirty videos worth of data
no_sequences = 30

# Capture duration per sample (seconds), independent of camera FPS
capture_seconds = 3.0
start_delay_ms = 3000

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
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 3. HELPER FUNCTION (Extract Keypoints)
# ==========================================
def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()

            if label == 'Left':
                lh = coords
            elif label == 'Right':
                rh = coords
    
    # Concatenate hands only (Total 126 points)
    return np.concatenate([lh, rh])


def clear_sequence_folder(action, sequence):
    seq_path = os.path.join(DATA_PATH, action, str(sequence))
    if not os.path.exists(seq_path):
        os.makedirs(seq_path)
        return

    for fname in os.listdir(seq_path):
        if fname.endswith('.npy'):
            os.remove(os.path.join(seq_path, fname))

# ==========================================
# 4. MAIN DATA COLLECTION LOOP
# ==========================================
cap = cv2.VideoCapture(0)

# Track hands only to remove face and arm/body processing
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    quit_requested = False
    
    # Loop through actions (hello -> thanks -> iloveyou)
    for action in actions:
        if quit_requested:
            break

        # Loop through sequences (videos) 0 to 29
        for sequence in range(no_sequences):
            clear_sequence_folder(action, sequence)

            # Warm-up screen before each sample
            ret, frame = cap.read()
            if not ret:
                continue

            warmup_image = frame.copy()
            cv2.putText(warmup_image, 'STARTING COLLECTION', (120, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(warmup_image, f'Action: {action} | Sample: {sequence}', (15, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(warmup_image, f'Preparing for {start_delay_ms // 1000}s...', (15, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', warmup_image)

            if cv2.waitKey(start_delay_ms) & 0xFF == ord('q'):
                quit_requested = True
                break

            frame_num = 0
            start_time = cv2.getTickCount() / cv2.getTickFrequency()

            while True:
                now = cv2.getTickCount() / cv2.getTickFrequency()
                elapsed = now - start_time

                if elapsed >= capture_seconds:
                    break

                # Read Feed
                ret, frame = cap.read()
                if not ret:
                    continue

                # Make detections
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)

                # Draw landmarks
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw Hands
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                remaining = max(0.0, capture_seconds - elapsed)
                cv2.putText(image, f'Collecting: {action} | Sample: {sequence}', (15, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Time left: {remaining:.1f}s', (15, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Frames captured: {frame_num}', (15, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                
                # EXPORT KEYPOINTS
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    quit_requested = True
                    break

                frame_num += 1

            if quit_requested:
                break
                    
    cap.release()
    cv2.destroyAllWindows()