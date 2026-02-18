
import tensorflow as tf
import os

try:
    print("Loading model...")
    model = tf.keras.models.load_model('model/best_model.keras')
    print("Model loaded. Converting to TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimizations for size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    output_path = 'model/best_model.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Conversion successful! Saved to {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
except Exception as e:
    print(f"Error: {e}")
