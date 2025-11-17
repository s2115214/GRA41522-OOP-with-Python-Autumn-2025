from tensorflow.keras import layers



class BiCoder(layers.Layer):
    def __init__(self, input_shape = None, latent_dim=50, filters=32, kernel_size=3, strides=2, activation='relu'):
        """Parent class to the encoder and decoder.
        Has shared hyperparameters and functionalities.

        Args:
            input_shape (samples, height, width, channels): The shape of the input image.
            latent_dim (int): Number of latent dimensions. Encoder uses it for reducing input to this dimension.
            Decoder uses it to define shape of the input.
            filters (int): How many filters used. Each layers uses this times some constant.
            kernel_size (int): kernel size used for each layer in the encoder and decoder.
            strides (int): strides used in the models. 
        """
        
        super().__init__()
        # shared hyperparameters. I use the same encoder/decoder as in neural_networks.py in the GRA-4152 repo
        # I just use the convolutional models as they work for both the bw and color images
        self._latent_dim  = latent_dim
        self._filters     = filters
        self._kernel_size = kernel_size
        self._strides     = strides
        self._activation  = activation
        self._input_shape = input_shape
        
        # all encoders/decoders will call self._build_network() to define architecture
        self.model = self._build_network()
    
    # child classes implement this
    def _build_network(self):
        raise NotImplementedError
    


    # Encoder overrides this method. Decoder inherits it
    def call(self, x):
        return self.model(x)