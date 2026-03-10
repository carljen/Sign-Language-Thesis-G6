import matplotlib
matplotlib.use('Agg') # Forces matplotlib to not create any windows
import os
import threading
import time
import tempfile

# Hide TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import TensorFlow first
import tensorflow as tf
from keras.models import load_model # Note: No 'tensorflow.' prefix



# Other imports
import cv2
import numpy as np
import mediapipe as mp
from deep_translator import GoogleTranslator
from gtts import gTTS
# from playsound import playsound


# ==========================================
# NORMALIZATION
# ==========================================
def normalize_to_nose(keypoints):
    norm = np.copy(keypoints)

    nx, ny, nz = norm[0], norm[1], norm[2]

    # If pose is not detected, return as-is
    if np.all(norm[:132] == 0):
        return norm

    # Pose: 33 points * 4 dims
    norm[0:132:4] -= nx
    norm[1:132:4] -= ny
    norm[2:132:4] -= nz

    # Left hand: 21 points * 3 dims
    if not np.all(norm[132:195] == 0):
        norm[132:195:3] -= nx
        norm[133:195:3] -= ny
        norm[134:195:3] -= nz

    # Right hand: 21 points * 3 dims
    if not np.all(norm[195:258] == 0):
        norm[195:258:3] -= nx
        norm[196:258:3] -= ny
        norm[197:258:3] -= nz

    return norm


# ==========================================
# EXTRACT KEYPOINTS
# ==========================================
def extract_keypoints(results):
    pose = (
        np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )

    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([pose, lh, rh])


# ==========================================
# TRANSLATION + SPEECH
# ==========================================
language_options = {
    "1": ("en", "English"),
    "2": ("tl", "Filipino"),
    "3": ("es", "Spanish"),
    "4": ("fr", "French"),
    "5": ("de", "German"),
}

current_lang_key = "1"

action_text_map = {
    "1": "Number 1",
    "2": "Number 2",
    "3": "Number 3",
    "a": "Letter A",
    "b": "Letter B",
    "c": "Letter C",
    "d": "Letter D",
    "e": "Letter E",
    "f": "Letter F",
    "g": "Letter G",
    "h": "Letter H",
    "i": "Letter I",
    "j": "Letter J",
    "k": "Letter K",
    "l": "Letter L",
    "m": "Letter M",
    "n": "Letter N",
    "o": "Letter O",
    "p": "Letter P",
    "q": "Letter Q",
    "r": "Letter R",
    "s": "Letter S",
    "t": "Letter T",
    "u": "Letter U",
    "v": "Letter V",
    "w": "Letter W",
    "x": "Letter X",
    "y": "Letter Y",
    "z": "Letter Z",
    "again": "Again",
    "deaf": "Deaf",
    "goodbye": "Goodbye",
    "hearing": "Hearing",
    "hello": "Hello",
    "How are you": "How are you",
    "iloveyou": "I love you",
    "Nice to meet you!": "Nice to meet you!",
    "please": "Please",
    "see_you_later": "See you later",
    "sorry": "Sorry",
    "thanks": "Thank you",
    "What`s your name": "What's your name",
    "what`s up!": "What's up!",
    "you": "You",
}

speech_lock = threading.Lock()
last_spoken_text = ""
last_spoken_time = 0
speech_cooldown = 2.0


def translate_text(text, target_lang="en"):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text


def speak_text(text, lang="en"):
    with speech_lock:
        temp_path = "temp_voice.mp3"
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(temp_path)
            # Use os.system to play it in the background
            os.system(f"mpg123 -q {temp_path} &") 
        except Exception as e:
            print("TTS error:", e)


def speak_async(text, lang="en"):
    thread = threading.Thread(target=speak_text, args=(text, lang), daemon=True)
    thread.start()


# ==========================================
# LOAD MODEL
# ==========================================
model = load_model("action.h5")

# Must match training order exactly
actions = np.array([
    "1", "2", "3", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "again", "deaf", "goodbye", "hearing", "hello", "How are you", "iloveyou", "Nice to meet you!", "please", "see_you_later", "sorry", "thanks", "What`s your name", "what`s up!", "you"
])


# ==========================================
# CONFIG
# ==========================================
OUTPUT_FPS = 15.0
MODEL_COMPLEXITY = 0
SEQUENCE_LENGTH = 30
STABLE_FRAMES = 15
THRESHOLD = 0.85
MAX_SENTENCE = 5


# ==========================================
# MEDIAPIPE + CAMERA
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_collection.avi", fourcc, OUTPUT_FPS, (width, height))

prev_time = time.time()


# ==========================================
# MAIN LOOP
# ==========================================
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=MODEL_COMPLEXITY,
    static_image_mode=False,
) as holistic:

    print("Controls:")
    print("1 = English")
    print("2 = Filipino")
    print("3 = Spanish")
    print("4 = French")
    print("5 = German")
    print("q = Quit")

    sequence = []
    sentence = []
    predictions = []
    
    # --- NEW: Translation Cache Variables ---
    last_preview_action = ""
    last_preview_lang = ""
    cached_translation = ""
    # -------------

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        start_time = time.time()

        # Detection
        image.flags.writeable = False
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)

        # Draw
        image.flags.writeable = True

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract + normalize keypoints
        keypoints = extract_keypoints(results)
        keypoints = normalize_to_nose(keypoints)

        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        current_action = ""
        confidence = 0.0
        translated_preview = ""

        if len(sequence) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

            best_idx = int(np.argmax(res))
            current_action = actions[best_idx]
            confidence = float(res[best_idx])

            predictions.append(best_idx)

            print(f"Action: {current_action} | Confidence: {confidence:.2f}")

            last_stable = predictions[-STABLE_FRAMES:]

            if len(last_stable) == STABLE_FRAMES and last_stable.count(best_idx) == STABLE_FRAMES:
                if confidence > THRESHOLD:
                    added_new_word = False

                    if len(sentence) == 0:
                        sentence.append(current_action)
                        added_new_word = True
                    elif current_action != sentence[-1]:
                        sentence.append(current_action)
                        added_new_word = True

                    if len(sentence) > MAX_SENTENCE:
                        sentence = sentence[-MAX_SENTENCE:]

                    if added_new_word:
                        now = time.time()
                        base_text = action_text_map.get(current_action, current_action)
                        lang_code = language_options[current_lang_key][0]

                        if base_text != last_spoken_text or (now - last_spoken_time) > speech_cooldown:
                            translated_text = translate_text(base_text, lang_code)
                            speak_async(translated_text, lang_code)

                            last_spoken_text = base_text
                            last_spoken_time = now

            # ==========================================
            # SMART TRANSLATED PREVIEW (CACHED)
            # ==========================================
            base_preview = action_text_map.get(current_action, current_action)
            lang_code = language_options[current_lang_key][0]
            
            # ONLY call the internet API if the word or the language actually changed
            if current_action != last_preview_action or current_lang_key != last_preview_lang:
                cached_translation = translate_text(base_preview, lang_code)
                
                # Update our memory
                last_preview_action = current_action
                last_preview_lang = current_lang_key
                
            translated_preview = cached_translation
            # ==========================================

        # ==========================================
        # UI
        # ==========================================
        # Top sentence bar
        cv2.rectangle(image, (0, 0), (width, 45), (245, 117, 16), -1)
        cv2.putText(
            image,
            " ".join(sentence),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Current language
        lang_name = language_options[current_lang_key][1]
        cv2.putText(
            image,
            f"Language: {lang_name}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Language controls
        cv2.putText(
            image,
            "1-English  2-Filipino  3-Spanish  4-French  5-German",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        # Live status
        if current_action:
            status_text = f"Live: {current_action} ({int(confidence * 100)}%)"
            text_color = (0, 255, 0) if confidence > THRESHOLD else (0, 0, 255)

            cv2.putText(
                image,
                status_text,
                (10, height - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
                cv2.LINE_AA,
            )

            if translated_preview:
                cv2.putText(
                    image,
                    f"Translated: {translated_preview}",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        cv2.putText(
            image,
            f"FPS: {int(fps)}",
            (width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Latency
        latency_ms = (time.time() - start_time) * 1000
        print(f"Latency: {latency_ms:.2f} ms")

        # Save + show
        out.write(image)
        cv2.imshow("Sign Language Translator", image)

        # Controls
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
            current_lang_key = chr(key)
            print("Language changed to:", language_options[current_lang_key][1])

cap.release()
out.release()
cv2.destroyAllWindows()