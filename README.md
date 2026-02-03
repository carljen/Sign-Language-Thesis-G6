# Sign Language Detection System (Thesis)

This project uses Computer Vision (MediaPipe) and Deep Learning (TensorFlow/Keras) to detect sign language gestures in real-time.

## 🛠️ Setup & Installation

**Prerequisites:** Python 3.10 (Recommended)

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Dependencies:**
    Run this command to install TensorFlow, MediaPipe, and OpenCV:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you get errors on Windows, you may need to install specific versions manually:*
    ```bash
    pip install "numpy<2.0" "protobuf==3.20.3"
    ```

---

## 📂 Project Structure

* `1_data_collection.py`: Script to record your hand/body movements.
* `2_train_model.py`: Script to train the AI using the collected data.
* `3_realtime_test.py`: Script to test the detection with the webcam.
* `MP_Data/`: Folder containing the collected Numpy arrays (The Dataset).
* `logs/`: Folder for TensorBoard training logs.
* `action.h5`: The saved trained model file.

---

## 🤖 How to Train the Model (Instructions for Teammates)

If you need to add new signs or retrain the model, follow these **exact steps** to ensure accuracy.

### Step 1: Collect Data
Run the collection script:
```bash
python 1_data_collection.py
To Add a New Action: Open the script and change actions = np.array(['hello', 'thanks']) to include your new word.

Accuracy Checklist (CRITICAL):

For Moving Signs (e.g., Hello): Start with hands in lap -> Perform sign -> Return hands to lap. (The "Sandwich" Rule).

For Static Letters (e.g., A, B): Hold the pose like a statue for the full recording.

Vary Position: Move your body slightly left/right or forward/back every 10 videos so the AI doesn't memorize the background.

Step 2: Train the Model
Once data is collected, run the training script:

Bash

python 2_train_model.py
Expected Output: You will see a progress bar for "Epochs".

Success: The training is done when Categorical Accuracy is close to 1.00 (100%).

Result: This will generate/overwrite the action.h5 file.

Step 3: Update the Test Script
If you added new actions in Step 1, you MUST update the 3_realtime_test.py file.

Find: actions = np.array(['hello', 'thanks'])

Update it to match the exact order used in the collection script.

⚡ Testing & Instruments (For Defense)
To validate the system efficiency, run 3_realtime_test.py.

1. Latency Test (Speed)
The terminal will print the Latency in milliseconds (ms) for every frame.

Goal: < 100ms for real-time feel.

Instrument: Software Profiler (Python time module).

2. Distance Accuracy Test
Use a tape measure to mark 0.5m, 1.0m, and 1.5m on the floor.

Stand at 0.5m. Perform the sign 10 times. Record success rate.

Move to 1.0m. Repeat.

Move to 1.5m. Repeat.

⚠️ Troubleshooting Common Errors
1. "ModuleNotFoundError: No module named numpy..."

Fix: Your NumPy version is too new. Run: pip install "numpy<2.0"

2. Program Freezes/Hangs on Startup

Fix: A conflict between TensorFlow and Protobuf. Run: pip install "protobuf==3.20.3"

3. "ValueError: Shapes (None, ...) are incompatible"

Fix: You changed the extract_keypoints function. Ensure all scripts (Collection, Train, Test) use the exact same keypoint extraction logic (e.g., if you removed Face Mesh in one, remove it in all).