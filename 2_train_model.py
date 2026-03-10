from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def normalize_to_nose(keypoints):
    # copy the array so we don't break the original
    norm = np.copy(keypoints)

    # Hands-only vector layout: left hand (0:63), right hand (63:126)
    left_hand = norm[:63]
    right_hand = norm[63:126]

    # if no hands are detected, keep all zeros
    if np.all(left_hand == 0) and np.all(right_hand == 0):
        return norm

    # anchor to first visible wrist to make signs camera-position invariant
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

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data') 


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

print("Actions:", actions)
print("Number of classes:", len(actions))

no_sequences = 30 
sequence_length = 60
feature_dim = 126
DYNAMIC_ACTIONS = {
    action for action in actions
    if action not in set(list("abcdefghijklmnopqrstuvwxyz") + ["1", "2", "3", "none", "idle"])
}
NON_SIGN_CLASSES = {"none", "idle"}
NON_SIGN_WEIGHT_SCALE = 0.45
SIGN_WEIGHT_SCALE = 1.15
LOW_CONF_CLASSES = {"a", "How are you", "Nice to meet you!"}
LOW_CONF_WEIGHT_SCALE = 1.35
# ---------------------


def temporal_resample(frames, target_length):
    indices = np.linspace(0, len(frames) - 1, target_length).round().astype(int)
    return frames[indices]


def coerce_feature_dim(frame, target_dim):
    vec = np.asarray(frame).reshape(-1)

    if vec.size == target_dim:
        return vec

    # Legacy samples may include pose+hands (258). Keep hands-only tail (126).
    if target_dim == 126 and vec.size == 258:
        return vec[-126:]

    if vec.size > target_dim:
        return vec[:target_dim]

    padded = np.zeros(target_dim, dtype=vec.dtype)
    padded[:vec.size] = vec
    return padded


def load_sequence_resampled(action, sequence_idx, target_length):
    seq_path = os.path.join(DATA_PATH, action, str(sequence_idx))

    frame_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
    if not frame_files:
        raise FileNotFoundError(f"No .npy frames found in {seq_path}")

    frame_files = sorted(frame_files, key=lambda x: int(os.path.splitext(x)[0]))
    frames = [
        coerce_feature_dim(np.load(os.path.join(seq_path, f)), feature_dim)
        for f in frame_files
    ]
    frames = np.array(frames)

    # If frame count differs, sample indices uniformly to fixed LSTM length.
    return temporal_resample(frames, target_length)


def temporal_speed_augment(frames, target_length, speed_factor):
    # speed_factor > 1.0 gives a faster timeline, < 1.0 gives a slower timeline.
    warped_len = max(2, int(round(len(frames) / speed_factor)))
    warped = temporal_resample(frames, warped_len)
    return temporal_resample(warped, target_length)


def build_window(frames, jitter_std=0.0):
    window = []
    for res in frames:
        norm = normalize_to_nose(res)
        if jitter_std > 0:
            noise = np.random.normal(0, jitter_std, norm.shape)
            norm = norm + noise
        window.append(norm)
    return window

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        sample_frames = load_sequence_resampled(action, sequence, sequence_length)
        window = build_window(sample_frames, jitter_std=0.0)
        window_jittered = build_window(sample_frames, jitter_std=0.003)

        # 3. Feed the REAL video to the training list
        sequences.append(window)
        labels.append(label_map[action])
        
        # 4. Feed the FAKE (Jittered) video to the training list
        sequences.append(window_jittered)
        labels.append(label_map[action])

        # 5. Phrase classes get temporal speed augmentation for motion robustness.
        if action in DYNAMIC_ACTIONS:
            fast_frames = temporal_speed_augment(sample_frames, sequence_length, speed_factor=1.2)
            slow_frames = temporal_speed_augment(sample_frames, sequence_length, speed_factor=0.8)

            sequences.append(build_window(fast_frames, jitter_std=0.0))
            labels.append(label_map[action])

            sequences.append(build_window(slow_frames, jitter_std=0.0))
            labels.append(label_map[action])

        # Extra augmentation for classes currently showing low confidence.
        if action in LOW_CONF_CLASSES:
            sequences.append(build_window(sample_frames, jitter_std=0.005))
            labels.append(label_map[action])

            if action in DYNAMIC_ACTIONS:
                faster_frames = temporal_speed_augment(sample_frames, sequence_length, speed_factor=1.35)
                slower_frames = temporal_speed_augment(sample_frames, sequence_length, speed_factor=0.7)

                sequences.append(build_window(faster_frames, jitter_std=0.002))
                labels.append(label_map[action])

                sequences.append(build_window(slower_frames, jitter_std=0.002))
                labels.append(label_map[action])

X = np.array(sequences)
y_int = np.array(labels)

# adding stratify=y forces an equal split for all words
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X, y_int, test_size=0.05, stratify=y_int
)

y_train = to_categorical(y_train_int, num_classes=len(actions)).astype(int)
y_test = to_categorical(y_test_int, num_classes=len(actions)).astype(int)

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_int),
    y=y_train_int,
)
class_weight = {
    int(cls): float(weight)
    for cls, weight in zip(np.unique(y_train_int), class_weights_arr)
}

# Prevent `none/idle` from dominating the decision boundary.
for action_name, idx in label_map.items():
    if idx not in class_weight:
        continue

    if action_name in NON_SIGN_CLASSES:
        class_weight[idx] *= NON_SIGN_WEIGHT_SCALE
    else:
        class_weight[idx] *= SIGN_WEIGHT_SCALE

    if action_name in LOW_CONF_CLASSES:
        class_weight[idx] *= LOW_CONF_WEIGHT_SCALE

print("Class weights:")
for action_name, idx in label_map.items():
    print(f"  {action_name}: {class_weight.get(idx, 1.0):.3f}")

# ==========================================
# 1. UPGRADED MODEL ARCHITECTURE
# ==========================================
model = Sequential()

# Start wide (128) to capture all 126 hand keypoints without losing detail
model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(sequence_length, feature_dim)))
model.add(Dropout(0.2)) # Randomly drop 20% of neurons to prevent memorization

# Expand to 256 to find complex patterns (like the difference between C and E)
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))

# Narrow back down
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))

# Dense layers act as the "Spatial Shape Detectors" for your static alphabet
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ==========================================
# 2. SMART TRAINING WITH EARLY STOPPING
# ==========================================
# Stop training if accuracy doesn't improve for 30 epochs, and restore the best weights
early_stop = EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=40,
    mode='max',
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    mode='max',
    factor=0.5,
    patience=8,
    min_lr=1e-5,
    verbose=1,
)

tb_callback = TensorBoard(log_dir='Logs')

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=300,
    callbacks=[tb_callback, early_stop, lr_scheduler],
    class_weight=class_weight,
)

# Increased max epochs to 1000, but EarlyStopping will likely cut it off around 200-400

model.save('action.h5')
print("Model Trained Successfully!")

# ==========================================
# GENERATE CONFUSION MATRIX FOR THESIS
# ==========================================
print("Generating Confusion Matrix...")

# 1. Get the model's predictions on the TEST set (the 5% of data it hasn't seen)
yhat = model.predict(X_test)

# 2. Convert predictions back to simple numbers (0, 1, 2, etc.)
ytrue = y_test_int.tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# 3. Create the raw mathematical matrix
cm = confusion_matrix(ytrue, yhat)

# 4. Plot it beautifully using Seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=actions, yticklabels=actions)

# 5. Add labels for your thesis documentation
plt.title('Sign Language Detection - Confusion Matrix', fontsize=16)
plt.ylabel('Actual Sign (Ground Truth)', fontsize=14)
plt.xlabel('Predicted Sign (AI Output)', fontsize=14)

# 6. Save the image to your folder
plt.tight_layout()
plt.savefig('confusion_matrix_results.png')
print("Saved as 'confusion_matrix_results.png'! Check your project folder.")

# Show the pop-up window
plt.show()