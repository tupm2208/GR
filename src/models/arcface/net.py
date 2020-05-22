import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.applications import MobileNetV2
from src.models.arcface.layers import BatchNormalization


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


def ArcFaceModel(size=None, channels=3, name='arcface_model', embd_shape=512, w_decay=5e-4):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = MobileNetV2(input_shape=x.shape[1:], include_top=False)(x)

    embeds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    return Model(inputs, embeds, name=name)


if __name__ == '__main__':
    model = ArcFaceModel(112)
    model.load_weights('/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/arc_mbv2_ccrop/e_8_b_40000.ckpt')
    model.save('/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/pb_model/')