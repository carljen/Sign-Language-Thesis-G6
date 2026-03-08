from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def normalize_to_nose(keypoints):
    # copy the array so we don't break the original
    norm = np.copy(keypoints)
    
    # the nose is the very first landmark (index 0=x, 1=y, 2=z)
    nx, ny, nz = norm[0], norm[1], norm[2]
    
    # if pose is not detected, don't do math on zeros
    if np.all(norm[:132] == 0):
        return norm
        
    # 1. shift pose (skip visibility indices)
    norm[0:132:4] -= nx # x
    norm[1:132:4] -= ny # y
    norm[2:132:4] -= nz # z
    
    # 2. shift left hand if it is on screen
    if not np.all(norm[132:195] == 0):
        norm[132:195:3] -= nx
        norm[133:195:3] -= ny
        norm[134:195:3] -= nz
        
    # 3. shift right hand if it is on screen
    if not np.all(norm[195:258] == 0):
        norm[195:258:3] -= nx
        norm[196:258:3] -= ny
        norm[197:258:3] -= nz
        
    return norm

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['1', '2', '3', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'again', 'deaf', 'goodbye', 'hearing', 'hello', 'How are you', 'iloveyou', 'Nice to meet you!', 'please', 'see_you_later', 'sorry', 'thanks', 'What`s your name', 'what`s up!', 'you'])
no_sequences = 30 
sequence_length = 30
# ---------------------

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        window_jittered = [] # Create a second list for the "fake" data
        
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            
            # 1. Normalize to the face (The fix from the previous step)
            res = normalize_to_nose(res) 
            
            # 2. Add the normal, fixed frame to the standard window
            window.append(res)
            
            # === THE JITTER TRICK ===
            # Generate random noise (0.01 means a 1% pixel shift, simulating a shaky hand)
            noise = np.random.normal(0, 0.01, res.shape) 
            res_jittered = res + noise
            window_jittered.append(res_jittered)
            
        # 3. Feed the REAL video to the training list
        sequences.append(window)
        labels.append(label_map[action])
        
        # 4. Feed the FAKE (Jittered) video to the training list
        sequences.append(window_jittered)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# adding stratify=y forces an equal split for all words
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

# ==========================================
# 1. UPGRADED MODEL ARCHITECTURE
# ==========================================
model = Sequential()

# Start wide (128) to capture all 258 data points without losing detail
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(Dropout(0.2)) # Randomly drop 20% of neurons to prevent memorization

# Expand to 256 to find complex patterns (like the difference between C and E)
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))

# Narrow back down
model.add(LSTM(128, return_sequences=False, activation='relu'))
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
early_stop = EarlyStopping(monitor='categorical_accuracy', patience=30, restore_best_weights=True)
tb_callback = TensorBoard(log_dir='Logs')

# Increased max epochs to 1000, but EarlyStopping will likely cut it off around 200-400
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback, early_stop])

model.save('action.h5')
print("Model Trained Successfully!")

# ==========================================
# GENERATE CONFUSION MATRIX FOR THESIS
# ==========================================
print("Generating Confusion Matrix...")

# 1. Get the model's predictions on the TEST set (the 5% of data it hasn't seen)
yhat = model.predict(X_test)

# 2. Convert predictions back to simple numbers (0, 1, 2, etc.)
ytrue = np.argmax(y_test, axis=1).tolist()
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