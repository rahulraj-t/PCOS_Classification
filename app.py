"""PCOS image classifier web app (Flask + Keras).

1. Put your trained model at model/pcos_model.keras
2. Check the settings below (class index, rescaling) match how you trained.
3. Run:  python app.py   ->  http://127.0.0.1:5000
"""
import os
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

# ---------------- settings you may need to change ----------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "pcos_model.keras")
AFFECTED_CLASS = 1     # which class index means "affected" (check your training labels)
THRESHOLD = 0.5        # probability cut-off for calling an image "affected"
RESCALE = 255.0        # pixels are divided by this. Use 1.0 if your model already
                       # contains a Rescaling/preprocessing layer.
DEFAULT_SIZE = (224, 224, 3)  # used only if the model's input size can't be read
MAX_UPLOAD_MB = 10
ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp"}
# -----------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

model = keras.models.load_model(MODEL_PATH)

shape = model.input_shape
if isinstance(shape, list):
    shape = shape[0]
try:
    HEIGHT, WIDTH, CHANNELS = (int(d) for d in shape[1:4])
except (TypeError, ValueError):
    HEIGHT, WIDTH, CHANNELS = DEFAULT_SIZE


def preprocess(file_storage):
    img = Image.open(file_storage.stream)
    img = img.convert("L" if CHANNELS == 1 else "RGB").resize((WIDTH, HEIGHT))
    arr = np.asarray(img, dtype="float32") / RESCALE
    if CHANNELS == 1:
        arr = arr[..., None]
    return arr[None, ...]


@app.route("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "message": "Choose an image to upload."}), 400
    if file.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return jsonify({"ok": False, "message": "Use a PNG, JPG, BMP or WEBP image."}), 400

    try:
        X = preprocess(file)
    except (UnidentifiedImageError, OSError):
        return jsonify({"ok": False, "message": "That file isn't a readable image."}), 400

    try:
        out = np.asarray(model.predict(X, verbose=0))
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Model error: {exc}"}), 500

    if out.ndim == 2 and out.shape[1] > 1:      # softmax: one value per class
        p_affected = float(out[0][AFFECTED_CLASS])
    else:                                       # sigmoid: probability of class 1
        p1 = float(out.reshape(-1)[0])
        p_affected = p1 if AFFECTED_CLASS == 1 else 1 - p1

    return jsonify({
        "ok": True,
        "affected": p_affected >= THRESHOLD,
        "probability": p_affected,
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False, "message": f"Image is larger than {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)