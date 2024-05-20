import tensorflow as tf


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def reconstruction_loss(x, recon_x, mean, log_var):
    RMSE = rmse(x, recon_x)
    KLD = -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mean, 2) - tf.exp(log_var))
    return (RMSE + KLD) / x.shape[0]
