import os
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow import keras
from tensorflow.keras import layers


# =====================================================
# Custom Layers
# =====================================================

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
            shape=(
                self.num_patches,
                self.embed_dim
            ),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):

        return inputs + self.position_embedding

    def get_config(self):

        config = super().get_config()

        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim
        })

        return config


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

        self.reshape = layers.Reshape(
            (1, 1, channels)
        )

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

        self.ffn = tf.keras.Sequential([
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


# =====================================================
# App Settings
# =====================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "WomenSafe_stage1_best(1).keras"
)

AFFECTED_CLASS = 0

THRESHOLD = 0.7
RESCALE = 255.0
DEFAULT_SIZE = (224, 224, 3)

MAX_UPLOAD_MB = 10

ALLOWED = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "webp"
}

# =====================================================
# Flask App
# =====================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_UPLOAD_MB * 1024 * 1024
)

print("Loading WomenSafe model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "SEBlock": SEBlock,
        "TransformerBlock": TransformerBlock,
        "PositionalEmbedding": PositionalEmbedding
    },
    compile=False,
    safe_mode=False
)


print("Model loaded successfully!")

shape = model.input_shape

if isinstance(shape, list):
    shape = shape[0]

try:
    HEIGHT, WIDTH, CHANNELS = (
        int(d) for d in shape[1:4]
    )
except (TypeError, ValueError):
    HEIGHT, WIDTH, CHANNELS = DEFAULT_SIZE


def preprocess(file_storage):

    img = Image.open(file_storage.stream)

    img = img.convert(
        "L" if CHANNELS == 1 else "RGB"
    ).resize(
        (WIDTH, HEIGHT)
    )

    arr = np.asarray(
        img,
        dtype="float32"
    ) / RESCALE

    if CHANNELS == 1:
        arr = arr[..., None]

    return arr[None, ...]


@app.route("/")
def index():
    return render_template(
        "index.html",
        max_mb=MAX_UPLOAD_MB
    )


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify({
            "ok": False,
            "message": "Choose an image."
        }), 400

    if (
        file.filename.rsplit(".", 1)[-1].lower()
        not in ALLOWED
    ):
        return jsonify({
            "ok": False,
            "message": "Unsupported image format."
        }), 400

    try:
        X = preprocess(file)

    except (
        UnidentifiedImageError,
        OSError
    ):
        return jsonify({
            "ok": False,
            "message": "Invalid image."
        }), 400

    try:
        out = np.asarray(
            model.predict(
                X,
                verbose=0
            )
        )

    except Exception as exc:
        return jsonify({
            "ok": False,
            "message": str(exc)
        }), 500

    p_affected = float(
        out[0][AFFECTED_CLASS]
    )

    return jsonify({
        "ok": True,
        "affected": p_affected >= THRESHOLD,
        "probability": round(
            p_affected,
            4
        )
    })


@app.errorhandler(413)
def too_large(_):

    return jsonify({
        "ok": False,
        "message": (
            f"Image exceeds "
            f"{MAX_UPLOAD_MB} MB."
        )
    }), 413


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
