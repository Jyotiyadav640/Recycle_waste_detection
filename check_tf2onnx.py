try:
    print("Start check")
    import tf2onnx
    print("Imported tf2onnx:", tf2onnx.__version__)
    import onnx
    print("Imported onnx:", onnx.__version__)
    
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    
    # Try dummy conversion
    print("Testing dummy conversion...")
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
    spec = (tf.TensorSpec((None, 5), tf.float32, name="input"),)
    path = "model/dummy.onnx"
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=path)
    print("Dummy conversion successful")
    
except Exception as e:
    import traceback
    traceback.print_exc()
