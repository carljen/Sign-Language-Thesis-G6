import matplotlib
matplotlib.use('Agg') # Forces matplotlib to not create any windows
import os
import threading
import time
import tempfile

# Hide TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import TensorFlow first
from tensorflow.keras.models import load_model

# Other imports
import cv2
import numpy as np
import mediapipe as mp
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound


# ==========================================
# NORMALIZATION
# ==========================================
def normalize_to_nose(keypoints):
    norm = np.copy(keypoints)

    # Hands-only vector layout: left hand (0:63), right hand (63:126)
    left_hand = norm[:63]
    right_hand = norm[63:126]

    if np.all(left_hand == 0) and np.all(right_hand == 0):
        return norm

    # Use the first visible wrist as translation anchor.
    if not np.all(left_hand == 0):
        ax, ay, az = left_hand[0], left_hand[1], left_hand[2]
    else:
        ax, ay, az = right_hand[0], right_hand[1], right_hand[2]

    if not np.all(left_hand == 0):
        norm[0:63:3] -= ax
        norm[1:63:3] -= ay
        norm[2:63:3] -= az

    if not np.all(right_hand == 0):
        norm[63:126:3] -= ax
        norm[64:126:3] -= ay
        norm[65:126:3] -= az

    return norm


def adapt_features_for_model(hand_keypoints, expected_features):
    # Keep detector hands-only; if model is legacy (258), prepend zero pose features.
    if expected_features == 126:
        return hand_keypoints

    if expected_features == 258:
        pose_zeros = np.zeros(33 * 4)
        return np.concatenate([pose_zeros, hand_keypoints])

    raise ValueError(
        f"Unsupported model input feature size: {expected_features}. Expected 126 or 258."
    )


# ==========================================
# EXTRACT KEYPOINTS
# ==========================================
def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if label == "Left":
                lh = coords
            elif label == "Right":
                rh = coords

    return np.concatenate([lh, rh])


# ==========================================
# HELPERS
# ==========================================
def calculate_motion(sequence):
    if len(sequence) < 2:
        return 0.0

    curr = sequence[-1]
    prev = sequence[-2]
    diff = np.abs(curr - prev)
    return float(np.mean(diff))


def has_enough_landmarks(results):
    left_ok = False
    right_ok = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for handedness in results.multi_handedness:
            label = handedness.classification[0].label
            if label == "Left":
                left_ok = True
            elif label == "Right":
                right_ok = True

    return left_ok or right_ok


def hand_visible(results):
    return bool(results.multi_hand_landmarks)


def is_non_sign(label):
    return label in ["none", "idle"]


# ==========================================
# TRANSLATION + SPEECH
# ==========================================
language_options = {
    "1": ("en", "English"),
    "2": ("tl", "Filipino"),
    "3": ("es", "Spanish"),
    "4": ("fr", "French"),
    "5": ("de", "German"),
    "6": ("it", "Italian"),
    "7": ("pt", "Portuguese"),
    "8": ("ru", "Russian"),
    "9": ("ja", "Japanese"),
    "0": ("ko", "Korean"),
    "a": ("ar", "Arabic"),
    "b": ("hi", "Hindi"),
    "c": ("zh-CN", "Chinese (Simplified)"),
    "d": ("ceb", "Cebuano"),
    "e": ("vi", "Vietnamese"),
    "f": ("th", "Thai"),
    "g": ("id", "Indonesian"),
    "h": ("ms", "Malay"),
    "i": ("tr", "Turkish"),
    "j": ("nl", "Dutch"),
    "k": ("pl", "Polish"),
    "l": ("sv", "Swedish"),
    "m": ("no", "Norwegian"),
    "n": ("da", "Danish"),
    "o": ("fi", "Finnish"),
    "p": ("cs", "Czech"),
    "r": ("el", "Greek"),
    "s": ("ro", "Romanian"),
    "t": ("uk", "Ukrainian"),
    "u": ("hu", "Hungarian"),
    "v": ("sr", "Serbian"),
    "w": ("hr", "Croatian"),
    "x": ("sk", "Slovak"),
    "y": ("sw", "Swahili"),
    "z": ("bn", "Bengali"),
}

current_lang_key = "1"

current_dialect_key = "std"

# Phrase-level overrides for dialect flavor. Fallback uses normal translator output.
dialect_phrase_map = {
    "ceb": {
        "Hello": "Kumusta",
        "How are you": "Kumusta ka",
        "What's your name": "Unsa imong ngalan",
        "What's up!": "Unsa may balita",
        "I love you": "Gihigugma tika",
        "Thank you": "Salamat",
        "Please": "Palihug",
        "Sorry": "Pasayloa ko",
        "Goodbye": "Babay",
        "See you later": "Magkita ta unya",
    },
}

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
translation_cache = {}


def translate_text(text, target_lang="en", dialect_key="std"):
    try:
        if target_lang == "en" and dialect_key == "std":
            return text

        cache_key = (text, target_lang, dialect_key)
        if cache_key in translation_cache:
            return translation_cache[cache_key]

        if dialect_key in dialect_phrase_map:
            if text in dialect_phrase_map[dialect_key]:
                translated = dialect_phrase_map[dialect_key][text]
                translation_cache[cache_key] = translated
                return translated

        if target_lang == "ceb":
            # Fallback translation target when Cebuano isn't directly supported by provider.
            target_lang = "tl"

        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        translation_cache[cache_key] = translated
        return translated
    except Exception as e:
        print("Translation error:", e)
        return text


def get_language_runtime(lang_key):
    lang_code, lang_name = language_options[lang_key]

    if lang_code == "ceb":
        return "tl", lang_name, "ceb"

    return lang_code, lang_name, "std"


def speak_text(base_text, lang="en", dialect_key="std"):
    with speech_lock:
        temp_path = None
        try:
            translated_text = translate_text(base_text, lang, dialect_key)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_path = temp_file.name

            tts = gTTS(text=translated_text, lang=lang)
            tts.save(temp_path)
            playsound(temp_path)
        except Exception as e:
            print("TTS error:", e)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


def speak_async(base_text, lang="en", dialect_key="std"):
    thread = threading.Thread(target=speak_text, args=(base_text, lang, dialect_key), daemon=True)
    thread.start()


def print_language_options():
    print("Language keys:")
    for key, (_, name) in language_options.items():
        print(f"  {key} = {name}")


def build_translated_sentence(sentence_actions, lang_code, dialect_key):
    if not sentence_actions:
        return ""

    translated_words = []
    for action in sentence_actions:
        base_text = action_text_map.get(action, action)
        translated_words.append(translate_text(base_text, lang_code, dialect_key))
    return " ".join(translated_words)


# ==========================================
# LOAD MODEL
# ==========================================
model = load_model("action.h5")
model_expected_timesteps = int(model.input_shape[1]) if model.input_shape[1] is not None else 60
model_expected_features = int(model.input_shape[-1])

if model_expected_features == 126:
    print("Model input detected: 126 features (hands-only model).")
elif model_expected_features == 258:
    print("Model input detected: 258 features (legacy model). Using zero-pose compatibility mode.")
else:
    raise ValueError(
        f"Unsupported model input shape {model.input_shape}. Use a model trained with 126 or 258 features."
    )

DATA_PATH = os.path.join("MP_Data")

def get_actions_from_data_path(data_path):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    action_dirs = [
        name for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    ]

    if not action_dirs:
        raise ValueError(f"No action folders found in: {data_path}")

    # Keep ordering deterministic so training and realtime map classes identically.
    return np.array(sorted(action_dirs, key=str.lower))


actions = get_actions_from_data_path(DATA_PATH)

DYNAMIC_ACTIONS = {
    action for action in actions
    if action not in set(list("abcdefghijklmnopqrstuvwxyz") + ["1", "2", "3", "none", "idle"])
}

print("Realtime actions:", actions)
print("Total classes:", len(actions))


# ==========================================
# CONFIG
# ==========================================
OUTPUT_FPS = 20.0
SEQUENCE_LENGTH = model_expected_timesteps
STABLE_FRAMES = 3
STATIC_THRESHOLD = 0.88
DYNAMIC_THRESHOLD = 0.78
STATIC_MARGIN_THRESHOLD = 0.22
DYNAMIC_MARGIN_THRESHOLD = 0.12
MIN_DYNAMIC_MOTION = 0.00025
NON_SIGN_THRESHOLD = 0.92
NON_SIGN_MARGIN_THRESHOLD = 0.28
NON_SIGN_STABLE_FRAMES = 6
MAX_SENTENCE = 5


# ==========================================
# MEDIAPIPE + CAMERA
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

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
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    print("Controls:")
    print_language_options()
    print("h = Show language list")
    print("q = Quit")

    sequence = []
    sentence = []
    predictions = []
    confidence_history = []
    margin_history = []
    motion_history = []
    non_sign_streak = 0

    last_preview_action = ""
    last_preview_lang = ""
    cached_translation = ""
    last_sentence_signature = None
    cached_sentence_display = ""

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        start_time = time.time()

        image.flags.writeable = False
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        keypoints = normalize_to_nose(keypoints)
        model_keypoints = adapt_features_for_model(keypoints, model_expected_features)

        sequence.append(model_keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        current_action = ""
        confidence = 0.0
        translated_preview = ""
        motion_score = 0.0
        quality_ok = False

        if len(sequence) == SEQUENCE_LENGTH and has_enough_landmarks(results):
            motion_score = calculate_motion(sequence)

            # For letters, allow prediction whenever a hand is visible.
            if not hand_visible(results):
                current_action = ""
                confidence = 0.0
                translated_preview = ""
                predictions.clear()
                non_sign_streak = 0
            else:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                top5_idx = np.argsort(res)[-5:][::-1]
                print("Top 5 predictions:")
                for idx in top5_idx:
                    print(f"  {actions[idx]}: {float(res[idx]):.4f}")

                best_idx = int(np.argmax(res))
                predicted_action = actions[best_idx]
                confidence = float(res[best_idx])
                second_best = float(np.partition(res, -2)[-2]) if len(res) > 1 else 0.0
                margin = confidence - second_best

                predictions.append(best_idx)
                confidence_history.append(confidence)
                margin_history.append(margin)
                motion_history.append(motion_score)

                predictions = predictions[-STABLE_FRAMES:]
                confidence_history = confidence_history[-STABLE_FRAMES:]
                margin_history = margin_history[-STABLE_FRAMES:]
                motion_history = motion_history[-STABLE_FRAMES:]

                print(
                    f"Action: {predicted_action} | Confidence: {confidence:.2f} | Motion: {motion_score:.5f}"
                )

                last_stable = predictions
                avg_conf = float(np.mean(confidence_history)) if confidence_history else 0.0
                avg_margin = float(np.mean(margin_history)) if margin_history else 0.0
                avg_motion = float(np.mean(motion_history)) if motion_history else 0.0

                if predicted_action in DYNAMIC_ACTIONS:
                    confidence_gate = DYNAMIC_THRESHOLD
                    margin_gate = DYNAMIC_MARGIN_THRESHOLD
                    motion_gate_ok = avg_motion >= MIN_DYNAMIC_MOTION
                else:
                    confidence_gate = STATIC_THRESHOLD
                    margin_gate = STATIC_MARGIN_THRESHOLD
                    motion_gate_ok = True

                quality_ok = (
                    (avg_conf >= confidence_gate)
                    and (avg_margin >= margin_gate)
                    and motion_gate_ok
                )

                # Ignore none/idle as output, but keep them as model classes
                if is_non_sign(predicted_action):
                    current_action = ""
                    translated_preview = ""

                    # Do not immediately reset after one idle hit; wait for a strong, stable streak.
                    is_strong_non_sign = (
                        confidence >= NON_SIGN_THRESHOLD and margin >= NON_SIGN_MARGIN_THRESHOLD
                    )
                    if is_strong_non_sign:
                        non_sign_streak += 1
                    else:
                        non_sign_streak = max(0, non_sign_streak - 1)

                    if non_sign_streak >= NON_SIGN_STABLE_FRAMES:
                        predictions.clear()
                        confidence_history.clear()
                        margin_history.clear()
                        motion_history.clear()
                else:
                    non_sign_streak = 0

                    # Only surface sign text when quality checks pass.
                    if quality_ok:
                        current_action = predicted_action
                    else:
                        current_action = ""
                        translated_preview = ""

                    if current_action:
                        base_preview = action_text_map.get(current_action, current_action)
                        lang_code, _, dialect_key = get_language_runtime(current_lang_key)

                        if current_action != last_preview_action or current_lang_key != last_preview_lang:
                            cached_translation = translate_text(
                                base_preview,
                                lang_code,
                                dialect_key,
                            )
                            last_preview_action = current_action
                            last_preview_lang = current_lang_key

                        translated_preview = cached_translation

                    if len(last_stable) == STABLE_FRAMES and last_stable.count(best_idx) == STABLE_FRAMES:
                        if quality_ok:
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
                                lang_code, _, dialect_key = get_language_runtime(current_lang_key)

                                if base_text != last_spoken_text or (now - last_spoken_time) > speech_cooldown:
                                    speak_async(base_text, lang_code, dialect_key)

                                    last_spoken_text = base_text
                                    last_spoken_time = now
        else:
            current_action = ""
            confidence = 0.0
            translated_preview = ""
            predictions.clear()
            confidence_history.clear()
            margin_history.clear()
            motion_history.clear()
            non_sign_streak = 0

        lang_code, lang_name, dialect_key = get_language_runtime(current_lang_key)
        sentence_signature = (tuple(sentence), lang_code, dialect_key)
        if sentence_signature != last_sentence_signature:
            cached_sentence_display = build_translated_sentence(sentence, lang_code, dialect_key)
            last_sentence_signature = sentence_signature

        # ==========================================
        # UI
        # ==========================================
        cv2.rectangle(image, (0, 0), (width, 45), (245, 117, 16), -1)
        cv2.putText(
            image,
            cached_sentence_display,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

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

        if dialect_key == "ceb":
            cv2.putText(
                image,
                "Dialect: Cebuano",
                (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 220, 120),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            image,
            "Keys: 1-0/a-z languages | h=list | q=quit",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        live_label = translate_text("Live", lang_code, dialect_key)
        unknown_idle_label = translate_text("Unknown/Idle", lang_code, dialect_key)

        if current_action:
            status_action_text = translate_text(
                action_text_map.get(current_action, current_action),
                lang_code,
                dialect_key,
            )
            status_text = f"{live_label}: {status_action_text} ({int(confidence * 100)}%)"
            text_color = (0, 255, 0)
        else:
            status_text = f"{live_label}: {unknown_idle_label}"
            text_color = (180, 180, 180)

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

        cv2.putText(
            image,
            f"Motion: {motion_score:.4f}",
            (width - 220, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

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

        latency_ms = (time.time() - start_time) * 1000
        print(f"Latency: {latency_ms:.2f} ms")

        out.write(image)
        cv2.imshow("Sign Language Translator", image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("h"):
            print_language_options()
        else:
            pressed = chr(key).lower() if key < 128 else ""
            if pressed in language_options:
                current_lang_key = pressed
                print("Language changed to:", language_options[current_lang_key][1])

cap.release()
out.release()
cv2.destroyAllWindows()