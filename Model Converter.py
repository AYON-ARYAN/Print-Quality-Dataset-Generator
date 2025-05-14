import tensorflow as tf

# Create a minimal model to test on Android
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile & train with dummy data
model.compile(optimizer='adam', loss='categorical_crossentropy')
# dummy train (1 batch)
model.fit(x=tf.random.normal([1, 224, 224, 3]), y=tf.keras.utils.to_categorical([1], 3), epochs=1)

# Convert to TFLite with compatibility settings
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Save model
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Compatible TFLite model saved as 'model.tflite'")
