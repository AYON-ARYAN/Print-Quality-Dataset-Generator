import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="/Volumes/BLACK_SHARK/APPS_MADE/Print quality dataset/model.tflite")
interpreter.allocate_tensors()
print("✅ Model loaded successfully.")
