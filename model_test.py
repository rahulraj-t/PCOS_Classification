
import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.num_patches, self.embed_dim),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position_embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim,
        })
        return config


@keras.saving.register_keras_serializable()
class SEBlock(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(max(channels // self.reduction, 1),
                                activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")
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
        config.update({"reduction": self.reduction})
        return config


@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=128, num_heads=4,
                 ff_dim=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.dropout1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation=keras.activations.gelu),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.norm1(inputs)
        attention_output = self.attention(x, x, training=training)
        x = inputs + self.dropout1(attention_output, training=training)
        y = self.norm2(x)
        y = self.ffn(y, training=training)
        return x + self.dropout2(y, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


@keras.saving.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=16, embedding_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.projection = layers.Conv2D(
            filters=embedding_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def call(self, images):
        x = self.projection(images)
        return tf.reshape(x, [tf.shape(x)[0], -1, self.embedding_dim])

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim,
        })
        return config


model_path = r"model\WomenSafe_stage1_best(1).keras"

model = keras.models.load_model(
    model_path,
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "SEBlock": SEBlock,
        "TransformerBlock": TransformerBlock,
        "PatchEmbedding": PatchEmbedding,
    },
    compile=False,
    safe_mode=True,
)

print("MODEL LOADED SUCCESSFULLY")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Total parameters:", model.count_params())

