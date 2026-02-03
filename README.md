
# 🤟 Sign Language Detection System (Thesis)

A real-time **Sign Language Detection System** using **Computer Vision (MediaPipe)** and **Deep Learning (TensorFlow/Keras)**.
This project detects sign language gestures through a webcam and classifies them using a trained neural network model.

---

## 🛠️ Setup & Installation

### Prerequisites

* **Python 3.10** (Recommended)

---

### 📥 Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

---

### 📦 Install Dependencies

Install all required libraries using:

```bash
pip install -r requirements.txt
```

⚠️ **Windows Fix (If Errors Occur)**
Manually install compatible versions:

```bash
pip install "numpy<2.0" "protobuf==3.20.3"
```

---

## 📂 Project Structure

```text
├── 1_data_collection.py     # Script for recording sign language data
├── 2_train_model.py         # Script for training the model
├── 3_realtime_test.py       # Real-time webcam testing
├── MP_Data/                 # Dataset (stored as NumPy arrays)
├── logs/                    # TensorBoard training logs
└── action.h5                # Trained model file
```

---

## 🤖 How to Train the Model (For Teammates)

Follow these steps **exactly** to ensure consistent and accurate results.

---

### 🧩 Step 1: Collect Data

Run the data collection script:

```bash
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

