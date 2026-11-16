"""Model zoo: a compact custom CNN and seven frozen ImageNet backbones.

Every model takes raw RGB pixels in [0, 255] and carries its own input
preprocessing as a layer, so each backbone receives the normalisation its
ImageNet weights were trained with. Augmentation layers sit inside the model
and are active only when ``training=True``. Dropout layers are
:class:`MCDropout`, which can be switched on at inference for Monte Carlo
dropout without also putting BatchNorm or augmentation into training mode.

Written against the Keras 3 API only, so it runs on the TensorFlow backend
(Kaggle) and the PyTorch backend (local CPU tests).
"""

from __future__ import annotations

import math

import keras
from keras import layers, ops, regularizers

PKG = "skinclf"

# name -> (constructor, preprocessing mode). Modes follow keras.applications:
#   tf: x/127.5 - 1, torch: ImageNet mean/std on x/255,
#   caffe: RGB->BGR then subtract ImageNet BGR means, none: model rescales itself.
BACKBONES = {
    "VGG16": (keras.applications.VGG16, "caffe"),
    "ResNet50": (keras.applications.ResNet50, "caffe"),
    "MobileNetV2": (keras.applications.MobileNetV2, "tf"),
    "EfficientNetB0": (keras.applications.EfficientNetB0, "none"),
    "DenseNet121": (keras.applications.DenseNet121, "torch"),
    "InceptionV3": (keras.applications.InceptionV3, "tf"),
    "Xception": (keras.applications.Xception, "tf"),
}

# Classifier heads as in the original study: (units, dropout) per hidden
# layer, and whether BatchNorm follows the first dense layer.
HEADS = {
    "VGG16": ([(512, 0.5), (256, 0.3)], False),
    "ResNet50": ([(512, 0.5), (256, 0.3)], True),
    "MobileNetV2": ([(512, 0.4)], False),
    "EfficientNetB0": ([(512, 0.5)], True),
    "DenseNet121": ([(512, 0.5), (256, 0.3)], False),
    "InceptionV3": ([(512, 0.5)], False),
    "Xception": ([(512, 0.5)], True),
}

MODEL_NAMES = ["CustomCNN", *BACKBONES]


@keras.saving.register_keras_serializable(package=PKG)
class ImagenetPreprocess(layers.Layer):
    """Map [0, 255] RGB input to what a given backbone expects."""

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        if mode not in {"scale01", "tf", "torch", "caffe", "none"}:
            raise ValueError(mode)
        self.mode = mode

    def call(self, x):
        x = ops.cast(x, "float32")
        if self.mode == "scale01":
            return x / 255.0
        if self.mode == "tf":
            return x / 127.5 - 1.0
        if self.mode == "torch":
            mean = ops.convert_to_tensor([0.485, 0.456, 0.406])
            std = ops.convert_to_tensor([0.229, 0.224, 0.225])
            return (x / 255.0 - mean) / std
        if self.mode == "caffe":
            x = ops.flip(x, axis=-1)
            return x - ops.convert_to_tensor([103.939, 116.779, 123.68])
        return x

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


@keras.saving.register_keras_serializable(package=PKG)
class MCDropout(layers.Dropout):
    """Dropout that stays active at inference while ``mc_active`` is set."""

    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.mc_active = False

    def call(self, inputs, training=False):
        return super().call(inputs, training=True if self.mc_active else training)


def set_mc_dropout(model: keras.Model, active: bool) -> None:
    for layer in _all_layers(model):
        if isinstance(layer, MCDropout):
            layer.mc_active = active


def _all_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from _all_layers(layer)


def augmentation(seed: int) -> keras.Sequential:
    """Rotation ±20°, shift ±20 %, zoom ±20 %, horizontal and vertical flips."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(20 / 360, fill_mode="nearest", seed=seed + 1),
            layers.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=seed + 2),
            layers.RandomZoom(0.2, fill_mode="nearest", seed=seed + 3),
        ],
        name="augmentation",
    )


def _dense(units, l2, name=None, activation=None, dtype=None):
    return layers.Dense(
        units,
        activation=activation,
        kernel_regularizer=regularizers.L2(l2) if l2 else None,
        name=name,
        dtype=dtype,
    )


def build_custom_cnn(x, l2: float):
    x = ImagenetPreprocess("scale01", name="preprocess")(x)
    for block, filters in enumerate([16, 32, 64, 192], start=1):
        for conv in (1, 2):
            x = layers.Conv2D(filters, 3, padding="same", name=f"block{block}_conv{conv}")(x)
            x = layers.LeakyReLU(negative_slope=0.1)(x)
        x = layers.MaxPooling2D(2, name=f"block{block}_pool")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = _dense(1536, l2, name="dense_1")(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = MCDropout(0.4)(x)
    return x


def build_transfer(name: str, x, img_size: int, l2: float, pretrained: bool):
    ctor, mode = BACKBONES[name]
    base = ctor(
        weights="imagenet" if pretrained else None,
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    x = ImagenetPreprocess(mode, name="preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    hidden, batchnorm = HEADS[name]
    for i, (units, rate) in enumerate(hidden):
        x = _dense(units, l2, activation="relu")(x)
        if batchnorm and i == 0:
            x = layers.BatchNormalization()(x)
        x = MCDropout(rate)(x)
    return x


def build_model(
    name: str,
    num_classes: int,
    img_size: int = 128,
    l2: float = 1e-4,
    pretrained: bool = True,
    augment: bool = True,
    seed: int = 0,
) -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation(seed)(inputs) if augment else inputs
    if name == "CustomCNN":
        x = build_custom_cnn(x, l2)
    elif name in BACKBONES:
        x = build_transfer(name, x, img_size, l2, pretrained)
    else:
        raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")
    outputs = _dense(num_classes, l2, activation="softmax", name="predictions", dtype="float32")(x)
    return keras.Model(inputs, outputs, name=name)


def count_params(model: keras.Model) -> tuple[int, int]:
    """(trainable, total) parameter counts; total includes frozen backbone weights."""
    trainable = sum(math.prod(w.shape) for w in model.trainable_weights)
    return int(trainable), int(model.count_params())
