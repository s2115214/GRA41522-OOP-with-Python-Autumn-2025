from tensorflow.keras import layers, Sequential
from bicoder import BiCoder
import numpy as np

class Decoder(BiCoder):
    def __init__(self,channel_out=3, target_shape=(4,4,128)):
        # Define hyperparameters specific to the Decoder, then inherit from BiCoder.
        self._channel_out = channel_out
        self._target_shape = target_shape
        self._units = np.prod(target_shape)
        super().__init__()

    # Decoder specific architecture. Used same structure as in neural_networks.py
    def _build_network(self):
        return Sequential(
                        [
                        layers.InputLayer(shape=(self._latent_dim,)),
                        layers.Dense(units=self._units, activation=self._activation),
                        layers.Reshape(target_shape=self._target_shape),
                        layers.Conv2DTranspose(
                            filters=self._filters*2, kernel_size=self._kernel_size, strides=self._strides, padding='same',output_padding=0,
                            activation=self._activation),
                        layers.Conv2DTranspose(
                            filters=self._filters, kernel_size=self._kernel_size, strides=self._strides, padding='same',output_padding=1,
                            activation=self._activation),
                        layers.Conv2DTranspose(
                            filters=self._channel_out, kernel_size=self._kernel_size, strides=self._strides, padding='same', output_padding=1),
                        layers.Activation('linear', dtype='float32'),
                        ]
                        )
        

