import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template_string
from skimage.feature import hog, local_binary_pattern

app = Flask(__name__)

# ==========================
# 1. Load Model (tanpa scaler)
# ==========================
MODEL_PATH = "model_sayur.pkl"

# Load model dengan joblib
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

# ==========================
# 2. Segmentasi GrabCut
# ==========================
def segment_grabcut(img):
    mask = np.zeros(img.shape[:2], np.uint8)

    h, w = img.shape[:2]
    rect = (int(w*0.15), int(h*0.15), int(w*0.7), int(h*0.7))

    bgModel = np.zeros((1,65), np.float64)
    fgModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    kernel = np.ones((5,5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

    return img * mask2[:, :, np.newaxis]

# ==========================
# 3. Ekstraksi Fitur (HOG + LBP)
# ==========================
def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        transform_sqrt=True
    )
    return hog_features

def extract_lbp_features(img, P=8, R=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    lbp = local_binary_pattern(gray, P, R, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2))

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

def extract_features(img):
    hog_feat = extract_hog_features(img)
    lbp_feat = extract_lbp_features(img)

    return np.hstack([hog_feat, lbp_feat])  # total = 8110 fitur

# ==========================
# 4. Preprocessing Gambar
# ==========================
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    segmented = segment_grabcut(img)

    feats = extract_features(segmented)
    return feats  # Tanpa scaling!

# ==========================
# 5. Mapping Label
# ==========================
label_order = ["bunga_kol", "cabai", "kubis", "sawi_hijau", "sawi_putih"]

pretty_names = {
    "bunga_kol": "Bunga Kol",
    "cabai": "Cabai",
    "kubis": "Kubis",
    "sawi_hijau": "Sawi Hijau",
    "sawi_putih": "Sawi Putih"
}

# ==========================
# 6. Template Halaman Web
# ==========================
HOME_HTML = """
<div style="text-align:center; padding:50px;">
    <h1>🥦 Vegetables Classifier 🍅</h1>
    <p>Unggah gambar sayur untuk diprediksi</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required><br><br>
        <button type="submit">Cek Gambar</button>
    </form>
</div>
"""

RESULT_HTML = """
<div style="text-align:center; padding:50px;">
    <h1>Hasil: {{ label }}</h1>
    <p>Confidence: {{ conf }}%</p>
    <a href="/">Coba Lagi</a>
</div>
"""

# ==========================
# 7. Routing Flask
# ==========================
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img_data = preprocess_image(file.read())

        # Prediksi dengan model sklearn (tanpa scaler)
        y_pred = model.predict([img_data])[0]
        y_proba = model.predict_proba([img_data])[0]

        class_name = label_order[y_pred]
        pretty = pretty_names[class_name]
        confidence = float(np.max(y_proba) * 100.0)

        return render_template_string(
            RESULT_HTML,
            label=pretty,
            conf=f"{confidence:.2f}"
        )

    except Exception as e:
        return f"<h2>Error: {e}</h2>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
