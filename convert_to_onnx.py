import tensorflow as tf
import tf2onnx
import os

model_path = 'model/best_model.keras'
output_path = 'model/best_model.onnx'

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    exit(1)

print(f"Loading model from {model_path}...")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    
    # Keras 3 compatibility fix:
    if not hasattr(model, 'output_names'):
        print("Patching model.output_names for Keras 3...")
        model.output_names = ['output_0'] 
    
    print("Converting to ONNX...")
    # Define input signature based on model input shape (150x150x3)
    spec = (tf.TensorSpec((None, 150, 150, 3), tf.float32, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    print(f"Conversion successful! Saved to {output_path}")
    
except Exception as e:
    print(f"Conversion failed: {e}")
    exit(1)
