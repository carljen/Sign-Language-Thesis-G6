import matplotlib
matplotlib.use('Agg') # Forces matplotlib to not create any windows
import os
import threading
import time
import tempfile

# Hide TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import TensorFlow
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from deep_translator import GoogleTranslator
from gtts import gTTS
import io
try:
    import pygame
    PYGAME_AVAILABLE = True
    # Initialize pygame mixer for audio
    pygame.mixer.init()
except ImportError:
    PYGAME_AVAILABLE = False


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

        # Try dialect phrases first (for custom translations)
        if dialect_key in dialect_phrase_map:
            if text in dialect_phrase_map[dialect_key]:
                translated = dialect_phrase_map[dialect_key][text]
                translation_cache[cache_key] = translated
                return translated

        # Use GoogleTranslator for any language
        # It supports 100+ languages via Google Translate API
        try:
            print(f"[TRANS] Translating '{text}' to {target_lang} ({dialect_key})")
            translated = GoogleTranslator(source="en", target=target_lang).translate(text)
            print(f"[TRANS] Result: '{translated}'")
            translation_cache[cache_key] = translated
            return translated
        except Exception as gtrans_error:
            print(f"GoogleTranslator error for {target_lang}: {gtrans_error}")
            # Fallback: return original text
            return text
            
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def get_language_runtime(lang_key):
    lang_code, lang_name = language_options[lang_key]

    if lang_code == "ceb":
        return "tl", lang_name, "ceb"

    return lang_code, lang_name, "std"


def speak_text(base_text, lang="en", dialect_key="std"):
    with speech_lock:
        try:
            translated_text = translate_text(base_text, lang, dialect_key)
            print(f"[TTS] {translated_text}")
            
            # gTTS language code mapping (gTTS uses different codes than GoogleTranslator)
            gtts_lang_map = {
                "en": "en",         # English
                "tl": "tl",         # Tagalog (closest to Filipino in gTTS)
                "fil": "tl",        # Filipino - use Tagalog
                "es": "es",         # Spanish
                "fr": "fr",         # French
                "de": "de",         # German
                "it": "it",         # Italian
                "pt": "pt",         # Portuguese
                "ru": "ru",         # Russian
                "ja": "ja",         # Japanese
                "ko": "ko",         # Korean
                "ar": "ar",         # Arabic
                "hi": "hi",         # Hindi
                "vi": "vi",         # Vietnamese
                "th": "th",         # Thai
                "ceb": "tl",        # Cebuano - use Tagalog
            }
            
            gtts_lang = gtts_lang_map.get(lang, "en")  # Default to English if not found
            
            # Use gTTS (Google Text-to-Speech)
            try:
                tts = gTTS(translated_text, lang=gtts_lang, slow=False)
                
                # Try to play with pygame (more reliable cross-platform)
                if PYGAME_AVAILABLE:
                    try:
                        # Save to BytesIO and play with pygame
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        
                        pygame.mixer.music.load(fp)
                        pygame.mixer.music.play()
                        
                        # Wait for audio to finish
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        
                        print(f"✓ TTS via gTTS ({gtts_lang})")
                    except Exception as pygame_err:
                        print(f"pygame playback error: {pygame_err}")
                        # Fallback: save to file
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                            temp_path = temp_audio.name
                            tts.save(temp_path)
                            print(f"Audio saved to: {temp_path}")
                            os.unlink(temp_path)
                else:
                    # Fallback: save to file
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                        temp_path = temp_audio.name
                        tts.save(temp_path)
                        print(f"Audio saved to: {temp_path}")
                        os.unlink(temp_path)
                        print("(install pygame for audio playback)")
            except Exception as e:
                print(f"gTTS error: {e}")
                
        except Exception as e:
            print(f"TTS error: {e}")


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
# LOAD TFLITE MODEL
# ==========================================
print("Loading TFLite Interpreter (Hardware Accelerated)...")
interpreter = tf.lite.Interpreter(model_path="action.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
model_expected_timesteps = input_details[0]['shape'][1] if input_details[0]['shape'][1] is not None else 60
model_expected_features = input_details[0]['shape'][-1]
actions = np.array(sorted([d for d in os.listdir("MP_Data") if os.path.isdir(os.path.join("MP_Data", d))], key=str.lower))

if model_expected_features == 126:
    print("Model input detected: 126 features (hands-only model).")
elif model_expected_features == 258:
    print("Model input detected: 258 features (legacy model). Using zero-pose compatibility mode.")
else:
    raise ValueError(
        f"Unsupported model input shape {input_details[0]['shape']}. Use a model trained with 126 or 258 features."
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
STATIC_THRESHOLD = 0.92          # Increased: 0.88 was too sensitive
DYNAMIC_THRESHOLD = 0.78
STATIC_MARGIN_THRESHOLD = 0.22
DYNAMIC_MARGIN_THRESHOLD = 0.12
MIN_DYNAMIC_MOTION = 0.00025
STATIC_MIN_MOTION = 0.0001       # Minimum: hand must be moving (not idle)
STATIC_MAX_MOTION = 0.004        # Maximum: too much motion = dynamic, not static
NON_SIGN_THRESHOLD = 0.92
NON_SIGN_MARGIN_THRESHOLD = 0.28
NON_SIGN_STABLE_FRAMES = 6
MAX_SENTENCE = 5


# ==========================================
# MEDIAPIPE + CAMERA (RPi5 Linux)
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# RPi5 Linux: Try V4L2 backend for USB cameras
cap = None
for i in range(10):  # Try indices 0-9 for RPi
    try:
        test_cap = cv2.VideoCapture(i, cv2.CAP_V4L2)  # V4L2 for Linux
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                cap = test_cap
                print(f"✓ Camera found at /dev/video{i}")
                break
            else:
                test_cap.release()
    except Exception as e:
        pass

if cap is None:
    print("Error: No camera found on /dev/video0-9")
    print("\nTroubleshooting for RPi5:")
    print("  1. Run: ls -la /dev/video*")
    print("  2. Check: sudo vcgencmd get_camera")
    print("  3. For USB camera: dmesg | grep -i usb")
    print("  4. Try: v4l2-ctl --list-devices")
    exit()

if not cap.isOpened():
    print("Error: Camera failed to initialize.")
    exit()

# RPi USB camera settings
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_collection.avi", fourcc, OUTPUT_FPS, (width, height))

prev_time = time.time()

# ==========================================
# TOUCH HANDLER FOR 7" SCREEN
# ==========================================
# Most common languages for easy access
TOUCH_LANGUAGES = [
    ("1", "English"), ("2", "Filipino"), ("3", "Spanish"),
    ("9", "Japanese"), ("0", "Korean"), ("a", "Arabic"),
    ("c", "Chinese"), ("f", "French"), ("d", "German"),
]

def touch_callback(event, x, y, flags, param):
    global current_lang_key
    if event == cv2.EVENT_LBUTTONDOWN:
        button_width = 100
        button_height = 40
        
        # Check which language button was clicked
        for idx, (key, name) in enumerate(TOUCH_LANGUAGES):
            bx = 10
            by = 40 + (idx * 45)  # Vertical spacing
            
            if bx < x < (bx + button_width) and by < y < (by + button_height):
                current_lang_key = key
                lang_code, lang_name = language_options[key]
                print(f"[TOUCH] Language switched to: {lang_name}")
                break

# Set up window with fullscreen and get screen resolution
cv2.namedWindow("Sign Language Translator", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Sign Language Translator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Sign Language Translator", touch_callback)

# Get screen resolution for stretching
import subprocess
try:
    # Try to get screen resolution from system
    result = subprocess.run(['xrandr'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'connected primary' in line or ' connected' in line:
            parts = line.split()
            for part in parts:
                if 'x' in part and '+' in part:
                    screen_width, screen_height = map(int, part.split('+')[0].split('x'))
                    print(f"Screen resolution: {screen_width}x{screen_height}")
                    break
except:
    # Fallback for RPi/other systems
    screen_width = 800   # Common RPi 7" screen
    screen_height = 480
    print(f"Using default screen resolution: {screen_width}x{screen_height}")

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
                # TFLite inference
                input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                res = interpreter.get_tensor(output_details[0]['index'])[0]

                # Bounds checking - ensure res size matches actions
                if len(res) != len(actions):
                    print(f"WARNING: Model output size {len(res)} != actions size {len(actions)}")
                    res = res[:len(actions)]  # Truncate if larger

                top5_count = min(5, len(res))
                top5_idx = np.argsort(res)[-top5_count:][::-1]
                print("Top 5 predictions:")
                for idx in top5_idx:
                    if 0 <= idx < len(actions):
                        print(f"  {actions[idx]}: {float(res[idx]):.4f}")

                best_idx = int(np.argmax(res))
                if best_idx >= len(actions):
                    best_idx = len(actions) - 1  # Safe fallback
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

                # ==========================================
                # MOTION-BASED STATIC vs DYNAMIC DISCRIMINATOR
                # ==========================================
                DYNAMIC_MOTION_THRESHOLD = 0.0005
                
                if predicted_action in DYNAMIC_ACTIONS:
                    action_type = "DYNAMIC"
                    confidence_gate = DYNAMIC_THRESHOLD
                    margin_gate = DYNAMIC_MARGIN_THRESHOLD
                    
                    # DYNAMIC ACTIONS MUST HAVE MOTION
                    if avg_motion < DYNAMIC_MOTION_THRESHOLD:
                        print(f"  [REJECT DYNAMIC] {predicted_action}: motion {avg_motion:.6f} < {DYNAMIC_MOTION_THRESHOLD:.6f} (too static)")
                        motion_gate_ok = False
                    else:
                        motion_gate_ok = True
                else:
                    action_type = "STATIC"
                    confidence_gate = STATIC_THRESHOLD
                    margin_gate = STATIC_MARGIN_THRESHOLD
                    
                    # STATIC LETTERS: must have moderate motion (not idle, not too dynamic)
                    if avg_motion < STATIC_MIN_MOTION:
                        print(f"  [REJECT STATIC] {predicted_action}: motion {avg_motion:.6f} < {STATIC_MIN_MOTION:.6f} (too idle)")
                        motion_gate_ok = False
                    elif avg_motion > STATIC_MAX_MOTION:
                        print(f"  [REJECT STATIC] {predicted_action}: motion {avg_motion:.6f} > {STATIC_MAX_MOTION:.6f} (too dynamic)")
                        motion_gate_ok = False
                    else:
                        motion_gate_ok = True

                quality_ok = (
                    (avg_conf >= confidence_gate)
                    and (avg_margin >= margin_gate)
                    and motion_gate_ok
                )

                # Debug: Show why detection passed/failed
                if predicted_action not in ["none", "idle"]:
                    conf_ok = avg_conf >= confidence_gate
                    margin_ok = avg_margin >= margin_gate
                    status = "✓ PASS" if (conf_ok and margin_ok and motion_gate_ok) else "✗ FAIL"
                    print(f"  {status} [{action_type}] {predicted_action} | Conf: {avg_conf:.2f}>{confidence_gate:.2f}? {conf_ok} | Margin: {avg_margin:.3f}>{margin_gate:.3f}? {margin_ok} | Motion: {avg_motion:.6f}")

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
        # UI - WITH 7" TOUCH SCREEN LANGUAGE BUTTONS
        # ==========================================
        cv2.rectangle(image, (0, 0), (width, 45), (245, 117, 16), -1)
        cv2.putText(
            image,
            cached_sentence_display,
            (120, 30),  # Shift right to avoid overlap with language buttons
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw language buttons on left side (for 7" touch screen) - TOUCHABLE
        button_width = 100
        button_height = 40
        
        for idx, (key, name) in enumerate(TOUCH_LANGUAGES):
            bx = 10
            by = 40 + (idx * 45)  # Vertical spacing
            
            # Highlight current language
            if current_lang_key == key:
                btn_color = (0, 200, 0)  # Green
            else:
                btn_color = (100, 100, 100)  # Gray
            
            cv2.rectangle(image, (bx, by), (bx + button_width, by + button_height), btn_color, -1)
            cv2.putText(
                image,
                name[:6],  # Truncate long names
                (bx + 5, by + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            image,
            f"Language: {lang_name}",
            (120, 75),
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
                (120, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 220, 120),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            image,
            "TAP language buttons | q=quit",
            (120, 105),
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
            (120, height - 55),
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
                (120, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            image,
            f"Motion: {motion_score:.4f}",
            (width - 220, 75),
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
            (width - 120, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        latency_ms = (time.time() - start_time) * 1000
        print(f"Latency: {latency_ms:.2f} ms")

        out.write(image)
        
        # Stretch image to fullscreen resolution
        display_image = cv2.resize(image, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Sign Language Translator", display_image)

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