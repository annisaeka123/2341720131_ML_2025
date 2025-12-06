import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from flask import Flask, request, render_template_string
from skimage.feature import hog

app = Flask(__name__)

# Load Model & Scaler
MODEL_PATH = 'day_night_model.h5'
SCALER_PATH = 'scaler.pkl'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ System Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading system: {e}")

def preprocess_image(image_bytes):
    # Decode gambar
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocessing (harus sama persis dengan training)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
                   
    return scaler.transform(hog_feat.reshape(1, -1))

@app.route('/', methods=['GET'])
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Day vs Night AI Classifier</title>
    </head>
    <body>
        <div style="text-align:center; padding:50px;">
            <h1>🌞 Day vs Night AI Classifier 🌙</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required><br><br>
                <button type="submit">Cek Gambar</button>
            </form>
        </div>
    </body>
    </html>
    """)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file', None)
        if file is None or file.filename == "":
            return "<h2 style='text-align:center'>Error: Tidak ada file yang diupload.</h2><center><a href='/'>Coba Lagi</a></center>"

        data = preprocess_image(file.read())
        pred = float(model.predict(data)[0][0])

        # asumsi output model = probabilitas "Day"
        if pred > 0.5:
            label = "Day (Siang)"
            confidence = pred * 100.0
        else:
            label = "Night (Malam)"
            confidence = (1.0 - pred) * 100.0

        return render_template_string(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hasil Prediksi</title>
        </head>
        <body>
            <div style="text-align:center; padding:50px;">
                <h1>Hasil: {label}</h1>
                <p>Confidence: {confidence:.2f}%</p>
                <a href='/'>Coba Lagi</a>
            </div>
        </body>
        </html>
        """)
    except Exception as e:
        return f"<h2 style='text-align:center'>Error: {e}</h2><center><a href='/'>Coba Lagi</a></center>"

if __name__ == '__main__':
    # Port 7860 wajib untuk Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)
