import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB5

from typing import Tuple, Callable, Any


class EfficientCellSeg(models.Model):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super(EfficientCellSeg, self).__init__()
        self._layers = {}
        self.input_layer = layers.Input(input_shape, name="input_resized")
        self.backbone = EfficientNetB5(
            input_tensor=self.input_layer,
            weights="imagenet",
            include_top=False
        )
        self.encoder = models.Model(
            inputs=self.backbone.input,
            outputs=self.backbone.get_layer("block6a_expand_activation").output
        )
        self.skip_connections = [
            "input_resized",
            "block1a_activation",
            "block3a_expand_activation",
            "block4a_expand_activation"
        ]

    def get(self, name: str, ctor: Callable[..., Any], *args, **kwargs) -> layers.Layer:
        """Helper func to skip initializing layers in the init method."""
        if name not in self._layers:
            self._layers[name] = ctor(*args, **kwargs)
        return self._layers[name]

    def call(self, img: np.ndarray) -> np.ndarray:
        x = self.encoder(img)

        x = self.get("up1", layers.UpSampling2D, interpolation="bilinear")(x)
        skip1 = self.encoder.get_layer(self.skip_connections[-1]).output
        x = self.get("concat1", layers.Concatenate)([x, skip1])
        x = self.get("conv1.1", layers.Conv2D, filters=64, kernel_size=3, padding="same")(x)
        x = self.get("bn1.1", layers.BatchNormalization)(x)
        x = self.get("relu1.1", layers.Activation, "relu")(x)
        x = self.get("conv1.2", layers.Conv2D, filters=64, kernel_size=3, padding="same")(x)
        x = self.get("bn1.2", layers.BatchNormalization)(x)
        x = self.get("relu1.2", layers.Activation, "relu")(x)

        x = self.get("up2", layers.UpSampling2D, interpolation="bilinear")(x)
        skip2 = self.encoder.get_layer(self.skip_connections[2]).output
        x = self.get("concat2", layers.Concatenate)([x, skip2])
        x = self.get("conv2.1", layers.Conv2D, filters=48, kernel_size=3, padding="same")(x)
        x = self.get("bn2.1", layers.BatchNormalization)(x)
        x = self.get("relu2.1", layers.Activation, "relu")(x)
        x = self.get("conv2.2", layers.Conv2D, filters=48, kernel_size=3, padding="same")(x)
        x = self.get("bn2.2", layers.BatchNormalization)(x)
        x = self.get("relu2.2", layers.Activation, "relu")(x)

        x = self.get("up3", layers.UpSampling2D, interpolation="bilinear")(x)
        skip3 = self.encoder.get_layer(self.skip_connections[1]).output
        x = self.get("concat3", layers.Concatenate)([x, skip3])
        x = self.get("conv3.1", layers.Conv2D, filters=32, kernel_size=3, padding="same")(x)
        x = self.get("bn3.1", layers.BatchNormalization)(x)
        x = self.get("relu3.1", layers.Activation, "relu")(x)
        x = self.get("conv3.2", layers.Conv2D, filters=32, kernel_size=3, padding="same")(x)
        x = self.get("bn3.2", layers.BatchNormalization)(x)
        x = self.get("relu3.2", layers.Activation, "relu")(x)

        x = self.get("up4", layers.UpSampling2D, interpolation="bilinear")(x)
        skip4 = self.encoder.get_layer(self.skip_connections[0]).output
        x = self.get("concat4", layers.Concatenate)([x, skip4])
        x = self.get("conv4.1", layers.Conv2D, filters=16, kernel_size=3, padding="same")(x)
        x = self.get("bn4.1", layers.BatchNormalization)(x)
        x = self.get("relu4.1", layers.Activation, "relu")(x)
        x = self.get("conv4.2", layers.Conv2D, filters=16, kernel_size=3, padding="same")(x)
        x = self.get("bn4.2", layers.BatchNormalization)(x)
        x = self.get("relu4.2", layers.Activation, "relu")(x)

        x = self.get("conv_last", layers.Conv2D, filters=1, kernel_size=1, padding="same")(x)
        x = self.get("sigmoid", layers.Activation, "sigmoid")(x)

        return x


def create_efficient_cell_seg(img_h: int = None, img_w: int = None, inner_h: int = 384,
                              inner_w: int = 384, imagenet_weights: bool = True,
                              num_filters_decoder: list = [64, 48, 32, 16]) -> keras.Model:
    """
    Returns an EfficientCellSeg model. The resizing of the inputs and outputs is part
    of the model, if img_h and img_w are None, resizing is done dynamically.
    """
    input = keras.Input(shape=(img_h, img_w, 3), name="input_img", dtype=tf.float32)
    input_resized = layers.Resizing(inner_h, inner_w, interpolation="bilinear",
                                    crop_to_aspect_ratio=False, name="input_resized")(input)
    # Encoder/ backbone:
    if imagenet_weights:
        encoder = EfficientNetB5(input_tensor=input_resized, weights="imagenet", include_top=False)
    else:
        encoder = EfficientNetB5(input_tensor=input_resized, include_top=False)

    for layer in encoder.layers:
        if not layer.name.startswith("input"):
            layer._name = "en_" + layer.name

    encoder_output = encoder.get_layer("en_block6a_expand_activation").output
    skip_connections = ["input_resized", "en_block1a_activation", "en_block3a_expand_activation",
                        "en_block4a_expand_activation"]

    # Decoder:
    x = encoder_output

    for idx, (skip_conn, num) in enumerate(zip(reversed(skip_connections),
                                               num_filters_decoder)):
        x_skip = encoder.get_layer(skip_conn).output
        x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, x_skip])

        x = layers.Conv2D(num, (3, 3), padding="same", name=f"de_conv2d_{idx}0")(x)
        x = layers.BatchNormalization(name=f"de_batchnorm{idx}0")(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(num, (3, 3), padding="same", name=f"de_conv2d_{idx}1")(x)
        x = layers.BatchNormalization(name=f"de_batchnorm{idx}1")(x)
        x = layers.Activation("relu")(x)

    x = layers.Conv2D(1, (1, 1), padding="same", name="de_conv2d_last")(x)
    decoder_output = layers.Activation("sigmoid")(x)
    decoder_output = tf.image.resize(decoder_output, tf.shape(input)[1:3])

    model = keras.Model(inputs=input, outputs=decoder_output)

    return model