import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from flask import jsonify

# from keras.preprocessing import image # Removed
import numpy as np
from PIL import Image
app = Flask(__name__, template_folder='templates', static_folder='app/static')
# from keras.models import load_model # Removed for Vercel size limit
import onnxruntime as ort

MODEL_PATH = 'model/best_model.onnx'
# Initialize ONNX Runtime Session
model_session = ort.InferenceSession(MODEL_PATH)
import tempfile

# Use /tmp for Vercel (read-only filesystem)
UPLOAD_FOLDER = '/tmp'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True) # /tmp always exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="Please upload an image.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess execution for ONNX (using Pillow & Numpy only)
    img = Image.open(file_path).resize((150, 150))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 150, 150, 3)
    img_array = img_array / 255.0
    
    # Run Inference using ONNX
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    
    # Ensure input is float32
    prediction = model_session.run([output_name], {input_name: img_array.astype(np.float32)})[0][0][0]

    if prediction > 0.5:
        result = "♻️ This is Recyclable Waste"
    else:
        result = "🌿 This is Organic Waste"

    image_url = f"/uploads/{file.filename}"

    # 🔥 IMPORTANT: show_proceed = True
    return render_template(
        'index.html',
        prediction=result,
        image_path=image_url,
        show_proceed=True
    )



@app.route('/api/predict', methods=['POST'])
def api_predict():
    print("🔥 API HIT RECEIVED")

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img = Image.open(file_path).resize((150, 150))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    print("📸 File received:", file.filename)

    # Run Inference using ONNX
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    prediction = model_session.run([output_name], {input_name: img_array.astype(np.float32)})[0][0][0]

    recyclable = prediction > 0.5

    return jsonify({
        "label": "Recyclable" if recyclable else "Organic",
        "recyclable": recyclable,
        "points": 20 if recyclable else 0,
        "imageUrl": f"/uploads/{file.filename}"
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)  # 🔥 Enable HTTP    S)
