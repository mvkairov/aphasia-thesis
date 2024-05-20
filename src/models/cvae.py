from common.tfutils import get_layer

from tensorflow import keras
import tensorflow as tf
ll = keras.layers


class Encoder(keras.Model):
    def __init__(self, enc_layers, input_dim, latent_dim, n_classes, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.concat = ll.Concatenate(axis=-1)
        self.ff = keras.Sequential(
            # keras.layers.InputLayer(input_shape=(input_dim + n_classes,))
        )
        for params in enc_layers:
            self.ff.add(get_layer(**params))
        
        self.mean = ll.Dense(latent_dim)
        self.log_var = ll.Dense(latent_dim)

    def call(self, x, c):
        x = self.concat([x, c])
        x = self.ff(x)

        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class Decoder(keras.Model):
    def __init__(self, dec_layers, input_dim, latent_dim, n_classes, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.concat = ll.Concatenate(axis=-1)
        self.ff = keras.Sequential(
            # keras.layers.InputLayer(input_shape=(latent_dim + n_classes))
        )
        for params in dec_layers:
            self.ff.add(get_layer(**params))
        self.ff.add(ll.Dense(input_dim))
        
    def call(self, z, c):
        z = self.concat([z, c])
        x = self.ff(z)
        return x


class CVAE(keras.Model):
    def __init__(self, enc_layers, dec_layers, input_dim, latent_dim, n_classes, *args, **kwargs):
        super(CVAE, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.encoder = Encoder(enc_layers, input_dim, latent_dim, n_classes)
        self.decoder = Decoder(dec_layers, input_dim, latent_dim, n_classes)

    def call(self, x, c):
        mean, log_var = self.encoder(x, c)
        z = self.reparametrize(mean, log_var)

        recon_x = self.decoder(z, c)
        return recon_x, mean, log_var

    def reparametrize(self, mean, log_var):
        std = tf.exp(log_var / 2)
        eps = tf.random.normal(std.shape)
        return mean + eps * std
    
    def encode(self, x, c):
        return self.encoder(x, c)
    
    def decode(self, x, c):
        return self.decoder(x, c)

