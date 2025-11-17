from tensorflow.keras import layers, Sequential
from bicoder import BiCoder


class Encoder(BiCoder):
    def __init__(self, input_shape=(28,28,3)): 
        # Encoder needs the input shape of the images
        super().__init__(input_shape=input_shape)


    # Model architecture for the encoder. Used same structure as in neural_networks.py
    def _build_network(self):
        return Sequential(
                        [
                        layers.InputLayer(shape=self._input_shape),
                        layers.Conv2D(
                          filters=self._filters, kernel_size=self._kernel_size, strides=self._strides, activation=self._activation, padding='same'),
                        layers.Conv2D(
                          filters=2*self._filters, kernel_size=self._kernel_size, strides=self._strides, activation=self._activation, padding='same'),
                        layers.Conv2D(
                          filters=4*self._filters, kernel_size=self._kernel_size, strides=self._strides, activation=self._activation, padding='same'),
                        layers.Flatten(),
                        layers.Dense(2*self._latent_dim)
                        ]
                        )

    def call(self, x):
        # Forward pass through the encoder to get mu and log_var
        x = super().call(x)
        # Extract mu and log_var from the output
        mu = x[:, :self._latent_dim]
        log_var = x[:, self._latent_dim:]
        # Return it for use in reparameterization trick in the VAE class
        return mu, log_var



