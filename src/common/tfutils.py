from tensorflow import keras
import tensorflow as tf
import sys

optim = keras.optimizers
ll = keras.layers

def get_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                device=gpus[0],
                logical_devices=[
                    tf.config.LogicalDeviceConfiguration(memory_limit=32000)
                ],
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
            
            
def get_optimizer(id, *args, **kwargs):
    # if sys.platform == "darwin":
    #     opt_dict = {
    #         "adam": optim.legacy.Adam,
    #         "sgd": optim.legacy.SGD,
    #         "adamw": optim.AdamW,
    #         # "nadam": optim.legacy.Nadam,
    #         "rmsprop": optim.legacy.RMSprop
    #     }
    # else:
    opt_dict = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
        "nadam": optim.Nadam,
        "rmsprop": optim.RMSprop
    }
    return opt_dict[id](*args, **kwargs)


def get_layer(id, *args, **kwargs):
    layermap = {
        "fc": ll.Dense,
        "d": ll.Dropout,
        "c1d": ll.Conv1D,
        "c2d": ll.Conv2D,
        "c3d": ll.Conv3D,
        "c1dt": ll.Conv1DTranspose,
        "c2dt": ll.Conv2DTranspose,
        "c3dt": ll.Conv3DTranspose,
        "bn": ll.BatchNormalization,
        "reg": ll.ActivityRegularization
    }
    return layermap[id](*args, **kwargs)