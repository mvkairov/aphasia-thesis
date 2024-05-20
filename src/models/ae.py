from common.tfutils import get_layer

from tensorflow import keras
ll = keras.layers


class Encoder(keras.Model):
    def __init__(self, enc_layers, input_dim, latent_dim, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim

        self.ff = keras.Sequential(
            keras.layers.InputLayer(input_shape=(input_dim,))
        )
        for params in enc_layers:
            self.ff.add(get_layer(**params))
        self.ff.add(ll.Dense(latent_dim))

    def call(self, x):
        return self.ff(x)


class Decoder(keras.Model):
    def __init__(self, dec_layers, input_dim, latent_dim, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim

        self.ff = keras.Sequential(
            keras.layers.InputLayer(input_shape=(latent_dim,))
        )
        for params in dec_layers:
            self.ff.add(get_layer(**params))
        self.ff.add(ll.Dense(input_dim))
        
    def call(self, z):
        return self.ff(z)


class AutoEncoder(keras.Model):
    def __init__(self, enc_layers, dec_layers, input_dim, latent_dim, *args, **kwargs):
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim

        self.encoder = Encoder(enc_layers, input_dim, latent_dim)
        self.decoder = Decoder(dec_layers, input_dim, latent_dim)

    def call(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x, c):
        return self.decoder(x)

