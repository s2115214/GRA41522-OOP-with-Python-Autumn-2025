import tensorflow as tf
import numpy as np


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        """Inherits from tf.keras.Model. This allows my VAE class to track 
        self.encoder and self.decoder as sub-models and their trainable variables automatically.
        This is essential for my train method to work.
        """
        super().__init__()
        # Store encoder and decoder
        self._encoder = encoder
        self._decoder = decoder
        # Defines the latent_dim of the VAE to be the latent_dim of the encoder.
        self._latent_dim = encoder._latent_dim
        self._std = 0.75

    @tf.function
    def call(self, x):
        # Takes input data x and returns the mu and log_var
        mu_enc, log_var = self._encoder.call(x)
        # Uses the reparameterization trick to sample z from the latent distribution
        z = self._reparameterize(mu_enc, log_var)
        # Uses the latent samples z to make reconstructions mu as done in neural_networks.py
        mu_dec = self._decoder.call(z)
        # Computes and returns the loss
        loss = self._compute_loss(x, mu_enc, mu_dec, log_var)
        return loss
            
        
    @tf.function
    def train(self, x, optimizer):
        # Training function is as given in utils.py in the GRA-4152 repo
        with tf.GradientTape() as tape:
            loss = self.call(x)
            self._vae_loss =  loss
            gradients = tape.gradient(self._vae_loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
    
    def get_latent_representation(self, x):
        # Given some input data x, does a forward pass through the encoder to get mu and log_var
        mu, log_var = self._encoder.call(x)
        return mu
    
    
    def generate_from_prior(self, n_samples=16):
        # Generate samples by drawing z from the prior p(z) = N(0, I)
        # and decoding them using the decoder.

        # Sample z from Gaussian
        z = tf.random.normal(shape=(n_samples, self._latent_dim))
        # Decode latent samples into images/data
        mu = self._decoder.call(z)
        return mu
    
    def generate_from_posterior(self, x):
        # Generate samples by encoding x to get the posterior distribution q(z|x)
        mu, log_var = self._encoder.call(x)
        z = self._reparameterize(mu, log_var)
        # The samples z are drawn from the posterior distribution. Pass them through the decoder to get reconstructions.
        # Returns the reconstructed images from the posterior distribution.
        return self._decoder.call(z)

    
    def _compute_loss(self, x, mu_enc, mu_dec, log_var):
        # Computes the total loss by using given equations.
        log_prob = self._log_diag_mvn(x, mu_dec, tf.math.log(self._std))
        recon_loss = -tf.reduce_mean(log_prob)

        # KL loss as given in utils.py in the GRA-4152 repo
        kl_loss = tf.reduce_mean(self._kl_divergence(mu_enc, log_var))
        
        # Returns the total loss
        return recon_loss + kl_loss

    @staticmethod
    def _reparameterize(mu, log_var):
        # Reparameterization trick: Used in the methods generated_from_posterior and call
        eps = tf.random.normal(shape=tf.shape(mu))
        return  mu + tf.exp(0.5 * log_var) * eps
    
    @staticmethod
    def _kl_divergence(mu, log_var):
        # KL divergence as given in utils.py in the GRA-4152 repo
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=-1) 
    
    @staticmethod
    def _log_diag_mvn(x, mu, log_sigma):
        sum_axes = tf.range(1, tf.rank(mu))
        k = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
        logp = - 0.5 * k * tf.math.log(2*np.pi) \
            - log_sigma \
            - 0.5*tf.reduce_sum(tf.square(x - mu)/tf.math.exp(2.*log_sigma),axis=sum_axes)
        return logp
    

