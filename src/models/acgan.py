from common.tfutils import get_layer

from tensorflow import keras
ll = keras.layers


class ACGenerator(keras.Model):
    def __init__(self, gen_layers, input_dim, latent_dim, n_classes, *args, **kwargs):
        super(ACGenerator, self).__init__(*args, **kwargs)

        self.concat = ll.Concatenate(axis=1)
        self.ff = keras.Sequential(
            # ll.Input(shape=(latent_dim + n_classes,))
        )
        for params in gen_layers:
            self.ff.add(get_layer(**params))
        self.ff.add(ll.Dense(input_dim, activation='sigmoid'))
    
    def call(self, z, label, training=False):
        out = self.concat([z, label])
        out = self.ff(out)
        return out


class ACDiscriminator(keras.Model):
    def __init__(self, dis_layers, input_dim, n_classes, *args, **kwargs):
        super(ACDiscriminator, self).__init__(*args, **kwargs)

        self.ff = keras.Sequential(
            # ll.Input(shape=(input_dim,))
        )
        for params in dis_layers:
            self.ff.add(get_layer(**params))

        self.fc_dis = ll.Dense(1, activation='sigmoid')
        self.fc_aux = ll.Dense(n_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.ff(x)
        dis = self.fc_dis(x)
        aux = self.fc_aux(x)
        return dis, aux


class ACGAN(keras.Model):
    def __init__(self, gen_layers, dis_layers, input_dim, latent_dim, n_classes, *args, **kwargs):
        super(ACGAN, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.generator = ACGenerator(gen_layers, input_dim, latent_dim, n_classes)
        self.discriminator = ACDiscriminator(dis_layers, input_dim, n_classes)

        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.dis_loss_tracker = keras.metrics.Mean(name="dis_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.dis_loss_tracker]

    def generate(self, z, label):
        return self.generator(z, label)
    
    def decode(self, z, label):
        return self.generator(z, label)
    
    def discriminate(self, x):
        return self.discriminator(x)

    def predict(self, x):
        _, aux = self.discriminator(x)
        return aux
