from data.tab import get_tabular_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import numpy as np


def get_dataset(test_size=0.3, *args, **kwargs):
    X, y, input_dim, n_classes = get_tabular_data(*args, **kwargs)
    if test_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y)
        y_train = label_binarize(y_train, classes=list(range(n_classes)))
        y_val = np.array(y_val, dtype=np.int32)
        return (X_train, y_train), (X_val, y_val), input_dim, n_classes
    else:
        y = label_binarize(y, classes=list(range(n_classes)))
        return (X_train, y_train), None, input_dim, n_classes
