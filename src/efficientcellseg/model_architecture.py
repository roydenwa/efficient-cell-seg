import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB5

from typing import Tuple, Callable, Any


def compact_get(module: tf.Module) -> tf.Module:
    def get(self, name: str, constructor: Callable[..., Any], *args, **kwargs) -> layers.Layer:
        """Helper func to skip initializing layers in the init method."""
        if name not in self._layers:
            self._layers[name] = constructor(*args, **kwargs, name=name)
        return self._layers[name]

    setattr(module, "get", get)
    return module


@compact_get
class EfficientCellSeg(models.Model):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (384, 384, 3),
        filters_decoder: Tuple[int, int, int, int] = (64, 48, 32, 16)
    ):
        super(EfficientCellSeg, self).__init__()
        self._layers = {}
        input_layer = layers.Input(input_shape, name="input_resized")
        backbone = EfficientNetB5(
            input_tensor=input_layer,
            weights="imagenet",
            include_top=False
        )
        self.encoder = models.Model(
            inputs=backbone.input,
            outputs=[
                backbone.get_layer("input_resized").output,
                backbone.get_layer("block1a_activation").output,
                backbone.get_layer("block3a_expand_activation").output,
                backbone.get_layer("block4a_expand_activation").output,
                backbone.get_layer("block6a_expand_activation").output
            ],
            name="encoder"
        )
        self.filters_decoder = filters_decoder

    def call(self, img: tf.Tensor) -> tf.Tensor:
        x = self.get("resize_input", layers.Resizing, 384, 384)(img)
        *skips, x = self.encoder(x)

        for i in range(1, 5):
            x = self.get(f"up{i}", layers.UpSampling2D, interpolation="bilinear")(x)
            x = self.get(f"concat{i}", layers.Concatenate)([x, skips[-i]])
            x = self.get(f"conv_block{i}", ConvBlock, filters=self.filters_decoder[i - 1])(x)

        x = self.get("conv_last", layers.Conv2D, filters=1, kernel_size=1, padding="same")(x)
        x = self.get("sigmoid", layers.Activation, "sigmoid")(x)
        x = self.get("resize_output", DynamicResizing)(x, img.shape[1], img.shape[2])

        return x


@compact_get
class ConvBlock(layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self._layers = {}

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.get("conv1", layers.Conv2D, filters=self.filters,
                     kernel_size=3, padding="same")(x)
        x = self.get("bn1", layers.BatchNormalization)(x)
        x = self.get("relu1", layers.Activation, "relu")(x)
        x = self.get("conv2", layers.Conv2D, filters=self.filters,
                     kernel_size=3, padding="same")(x)
        x = self.get("bn2", layers.BatchNormalization)(x)
        x = self.get("relu2", layers.Activation, "relu")(x)

        return x


class DynamicResizing(layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicResizing, self).__init__(**kwargs)

    def call(self, x: tf.Tensor, height: int, width: int) -> tf.Tensor:
        x = tf.image.resize(x, size=(height, width))
        return x


def create_efficient_cell_seg(img_h: int = None, img_w: int = None, inner_h: int = 384,
                              inner_w: int = 384, imagenet_weights: bool = True,
                              num_filters_decoder: list = [64, 48, 32, 16]) -> keras.Model:
    """
    Returns an EfficientCellSeg model. The resizing of the inputs and outputs is part
    of the model, if img_h and img_w are None, resizing is done dynamically.
    """
    input = layers.Input(shape=(img_h, img_w, 3), name="input_img", dtype=tf.float32)
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

    model = models.Model(inputs=input, outputs=decoder_output)

    return model