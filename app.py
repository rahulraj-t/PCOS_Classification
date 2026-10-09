"""
WomenSafe PCOS Image Classifier
Flask + Keras

Run:
    python app.py

Open:
    http://127.0.0.1:5000

Model:
    model/WomenSafe_stage1_best(1).keras
"""

import os
import numpy as np
import tensorflow as tf
import keras
from chatbot import chat_bp
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from keras import layers


# =========================================================
# Custom Layers
# These must match the definitions used during training.
# =========================================================

@keras.saving.register_keras_serializable()
class PatchEmbedding(layers.Layer):

    def __init__(
        self,
        patch_size=16,
        embedding_dim=128,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.projection = layers.Conv2D(
            filters=embedding_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid"
        )

    def call(self, images):
        x = self.projection(images)

        batch_size = tf.shape(x)[0]

        x = tf.reshape(
            x,
            [batch_size, -1, self.embedding_dim]
        )

        return x

    def get_config(self):
        config = super().get_config()

        config.update({
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim
        })

        return config


@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):

    def __init__(
        self,
        num_patches,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.num_patches, self.embed_dim),
            initializer="random_normal",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position_embedding

    def get_config(self):
        config = super().get_config()

        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim
        })

        return config


@keras.saving.register_keras_serializable()
class SEBlock(layers.Layer):

    def __init__(
        self,
        reduction=16,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.reduction = reduction

    def build(self, input_shape):
        channels = int(input_shape[-1])

        self.gap = layers.GlobalAveragePooling2D()

        self.fc1 = layers.Dense(
            max(channels // self.reduction, 1),
            activation="relu"
        )

        self.fc2 = layers.Dense(
            channels,
            activation="sigmoid"
        )

        self.reshape = layers.Reshape((1, 1, channels))

        super().build(input_shape)

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)

        return inputs * x

    def get_config(self):
        config = super().get_config()

        config.update({
            "reduction": self.reduction
        })

        return config


@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.dropout1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.ffn = keras.Sequential([
            layers.Dense(
                ff_dim,
                activation=tf.keras.activations.gelu
            ),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])

        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.norm1(inputs)

        attention_output = self.attention(
            x,
            x,
            training=training
        )

        x = inputs + self.dropout1(
            attention_output,
            training=training
        )

        y = self.norm2(x)

        y = self.ffn(
            y,
            training=training
        )

        x = x + self.dropout2(
            y,
            training=training
        )

        return x

    def get_config(self):
        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout
        })

        return config


# =========================================================
# Application Settings
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "WomenSafe_stage1_best(1).keras"
)

# Training labels previously used:
# 0 = infected / PCOS affected
# 1 = noninfected / PCOS not affected
AFFECTED_CLASS = 0

THRESHOLD = 0.7
RESCALE = 255.0
DEFAULT_SIZE = (224, 224, 3)

MAX_UPLOAD_MB = 10

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "webp"
}


# =========================================================
# Flask Application
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_UPLOAD_MB * 1024 * 1024
)
app.register_blueprint(chat_bp)

# =========================================================
# Load Trained Model
# =========================================================

print("Loading WomenSafe model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PatchEmbedding": PatchEmbedding,
        "PositionalEmbedding": PositionalEmbedding,
        "SEBlock": SEBlock,
        "TransformerBlock": TransformerBlock
    },
    compile=False,
    safe_mode=True
)

print("WomenSafe model loaded successfully!")

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# =========================================================
# Read Model Input Shape
# =========================================================

shape = model.input_shape

if isinstance(shape, list):
    shape = shape[0]

try:
    HEIGHT, WIDTH, CHANNELS = (
        int(d) for d in shape[1:4]
    )
except (TypeError, ValueError):
    HEIGHT, WIDTH, CHANNELS = DEFAULT_SIZE


# =========================================================
# Image Preprocessing
# =========================================================

def preprocess(file_storage):
    """Read, resize and normalize an uploaded image."""

    with Image.open(file_storage.stream) as image:
        image = image.convert(
            "L" if CHANNELS == 1 else "RGB"
        )

        image = image.resize((WIDTH, HEIGHT))

        arr = np.asarray(
            image,
            dtype=np.float32
        ) / RESCALE

    if CHANNELS == 1:
        arr = arr[..., np.newaxis]

    return np.expand_dims(arr, axis=0)


# =========================================================
# Routes
# =========================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        max_mb=MAX_UPLOAD_MB
    )


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("image")

    # Check upload
    if file is None or not file.filename:
        return jsonify({
            "ok": False,
            "message": "Choose an image to upload."
        }), 400

    # Check extension
    extension = (
        file.filename.rsplit(".", 1)[-1].lower()
        if "." in file.filename
        else ""
    )

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "ok": False,
            "message": (
                "Use a PNG, JPG, JPEG, BMP or WEBP image."
            )
        }), 400

    # Preprocess image
    try:
        X = preprocess(file)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        return jsonify({
            "ok": False,
            "message": "That file isn't a readable image."
        }), 400

    # Run prediction
    try:
        output = np.asarray(
            model.predict(X, verbose=0)
        )

        if output.ndim == 2 and output.shape[1] > 1:
            # Multiclass / softmax output
            if not 0 <= AFFECTED_CLASS < output.shape[1]:
                raise ValueError(
                    "AFFECTED_CLASS does not match model outputs."
                )

            probability = float(
                output[0, AFFECTED_CLASS]
            )

        else:
            # Single-output sigmoid model
            p1 = float(output.reshape(-1)[0])

            probability = (
                p1 if AFFECTED_CLASS == 1
                else 1.0 - p1
            )

        if not np.isfinite(probability):
            raise ValueError(
                "The model returned an invalid probability."
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "The model output is outside the probability range."
            )

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "ok": False,
            "message": (
                "Prediction failed. Check the server logs."
            )
        }), 500

    affected = probability >= THRESHOLD

    return jsonify({
        "ok": True,
        "affected": affected,
        "probability": round(probability, 4)
    })


# =========================================================
# Error Handlers
# =========================================================

@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "ok": False,
        "message": (
            f"Image is larger than {MAX_UPLOAD_MB} MB."
        )
    }), 413


# =========================================================
# Run Application
# =========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

