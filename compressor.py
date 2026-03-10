import tensorflow as tf

print("Loading massive .h5 model...")
model = tf.keras.models.load_model('action.h5')

print("Setting up TFLite hardware compression...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- THE FIX: Allow complex LSTM math ops ---
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
# --------------------------------------------

converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Compressing... (This might take a minute)")
tflite_model = converter.convert()

with open('action.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! action.tflite is ready for the Raspberry Pi.")