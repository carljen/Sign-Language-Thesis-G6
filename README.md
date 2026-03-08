Sign Language Detection System (Thesis)
This project uses Computer Vision (MediaPipe) and Deep Learning (TensorFlow/Keras) to detect sign language gestures in real-time.

🛠️ Setup & Installation
Prerequisites: Python 3.10 (Recommended)

Clone the Repository:

Bash

git clone <repository_url>
cd <repository_name>
Install Dependencies:
Run this command to install TensorFlow, MediaPipe, and OpenCV:

Bash

pip install -r requirements.txt
Note: If you get errors on Windows, you may need to install specific versions manually:

Bash

pip install "numpy<2.0" "protobuf==3.20.3"
📂 Project Structure
1_data_collection.py: Script to record your hand/body movements.

2_train_model.py: Script to train the AI using the collected data.

3_realtime_test.py: Script to test the detection with the webcam.

MP_Data/: Folder containing the collected Numpy arrays (The Dataset).

logs/: Folder for TensorBoard training logs.

action.h5: The saved trained model file.

🤖 How to Train the Model (Instructions for Teammates)
If you need to add new signs or retrain the model, follow these exact steps to ensure accuracy.

Step 1: Collect Data
Run the collection script:

Bash

python 1_data_collection.py
```

#### ➕ Adding a New Action

Inside `1_data_collection.py`, locate:

```python
actions = np.array(['hello', 'thanks'])
```

Add your new sign here **in the exact order you want**.

---

#### ✅ Accuracy Checklist (CRITICAL)

**For Moving Signs (e.g., “Hello”):**

* Start with hands in lap
* Perform the sign
* Return hands to lap
  👉 *The “Sandwich Rule”*

**For Static Signs (e.g., A, B):**

* Hold the pose steadily for the entire recording duration

**Data Variation:**

* Slightly change body position (left/right or forward/back)
* Do this every ~10 recordings to avoid background memorization

---

### 🧠 Step 2: Train the Model

After data collection:

```bash
python 2_train_model.py
```

**Expected Output:**

* Epoch progress bar in the terminal

**Success Indicator:**

* `Categorical Accuracy` close to **1.00 (100%)**

**Result:**

* Generates or overwrites `action.h5`

---

### 🔁 Step 3: Update the Test Script

If you added new actions, update `3_realtime_test.py`.

Find:

```python
actions = np.array(['hello', 'thanks'])
```

Update it to **match the exact order** used in `1_data_collection.py`.

⚠️ Order mismatch = wrong predictions.

---

## ⚡ Testing & Instruments (For Defense)

Run:

```bash
python 3_realtime_test.py
```

---

### ⏱️ 1. Latency Test (Speed)

* Latency (in milliseconds) is printed per frame in the terminal

**Target:**

* `< 100 ms` for real-time performance

**Instrument Used:**

* Python `time` module (Software Profiler)

---

### 📏 2. Distance Accuracy Test

1. Mark distances using a tape measure:

   * 0.5 m
   * 1.0 m
   * 1.5 m

2. At each distance:

   * Perform the sign **10 times**
   * Record successful detections

3. Compare accuracy across distances

---

## ⚠️ Troubleshooting Common Errors

### ❌ `ModuleNotFoundError: No module named 'numpy'`

**Cause:** NumPy version is too new
**Fix:**

```bash
pip install "numpy<2.0"
```

---

### ❌ Program Freezes / Hangs on Startup

**Cause:** TensorFlow–Protobuf conflict
**Fix:**

```bash
pip install "protobuf==3.20.3"
```

---

### ❌ `ValueError: Shapes (None, ...) are incompatible`

**Cause:** Inconsistent `extract_keypoints()` logic across scripts

**Fix:**

* Ensure **Data Collection**, **Training**, and **Testing** scripts use the **same keypoint extraction**
* If you removed Face Mesh or Body landmarks, remove them in **all scripts**

---

