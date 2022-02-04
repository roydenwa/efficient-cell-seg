import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB5


def create_efficient_cell_seg(img_h: int = None, img_w: int = None, inner_h: int = 384,
                              inner_w: int = 384, imagenet_weights: bool = True,
                              num_filters_decoder: list = [64, 48, 32, 16]) -> keras.Model:
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