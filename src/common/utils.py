from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix
from tqdm import tqdm, tqdm_notebook
from matplotlib import pyplot as plt
import numpy as np

from datetime import datetime
import pickle
import PIL
import sys


def make_name(cfg):
    run_name = cfg.ds.matter
    if cfg.ds.get("lesions", False):
        run_name += "-l"
    if cfg.ds.get("demo", False):
        run_name += "-d"
    if cfg.ds.get("zscale", False):
        run_name += "-z"
    if cfg.ds.get("minmax", False):
        run_name += "-m"
    if cfg.ds.get("gram", False):
        run_name += "-g"
    run_name += f"-{cfg.model_id}-{cfg.ds.method}-"
    if cfg.ds.get("target", False):
        run_name += f"{cfg.ds.target}-"
    run_name += datetime.now().strftime("%H:%M-%d-%m")
    return run_name


def make_msg(test_name=None, hyperopt_results=None, cv_results=None):
    msg = ""
    if test_name is not None:
        msg = f"{test_name}.json\n\n"
    if hyperopt_results is not None:
        for name, value in hyperopt_results.items():
            msg += f"{name} = {value}\n"
        msg += "\n"
    if cv_results is not None:
        for name, (mean, std) in cv_results.items():
            msg += f"{name}: {mean:.5f} ± {std:.5f}\n"
    return msg


def save_to_pkl(obj, name, save_path):
    with open(f"{save_path}/{name}.pkl", "wb") as file:
        pickle.dump(obj, file)


def reg_to_clf_target(y, target, n_classes=None):
    def map_asa(val):
        if val > 260:
            return 0
        elif val > 220:
            return 1
        elif val > 175:
            return 2
        elif val > 130:
            return 3
        elif val > 90:
            return 4
        else:
            return 5
    
    if target == "asa":
        return np.vectorize(map_asa, otypes=[np.int32])(y)
    elif n_classes is not None:
        return np.round(np.clip(y, 0, n_classes - 1)).astype(np.int32)
    else:
        raise ValueError


class RegAsClfStratifiedKFold(StratifiedKFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None, target, n_classes):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)
        self.target = target
        self.n_classes = n_classes

    def split(self, X, y, groups=None):
        y_copy = np.copy(y)
        y_copy = reg_to_clf_target(y_copy, self.target, self.n_classes)
        return super().split(X, y_copy, groups)


def get_cv(method, *args, reg_as_clf=False, target=None, n_classes=None, **kwargs):
    if method == "clf":
        return StratifiedKFold(*args, **kwargs)
    elif reg_as_clf:
        return RegAsClfStratifiedKFold(*args, **kwargs, target=target, n_classes=n_classes)
    else:
        return KFold(*args, **kwargs)


def plot_confusion_matrix(cm, target_names, subscript=None):
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Purples"))
    plt.title("Classification confusion matrix")
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, ha="right")
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    if subscript is not None:
        plt.xlabel(subscript)
    return fig


def get_dict_of_lists(d, return_mean_std=False):
    res = {}
    for key in d[0].keys():
        key_values = [x[key] for x in d]
        if return_mean_std:
            res[key] = (np.mean(key_values), np.std(key_values))
        else:
            res[key] = key_values
    return res


def sklearn_best_params(search_results):
    search_params = search_results["params"][0].keys()
    best_comb = np.argmax(search_results["mean_test_score"])
    best_params = search_results["params"][best_comb]
    best_dict = {}
    for param in search_params:
        best_dict[param.removeprefix("model__")] = best_params[param]
    return best_dict


def make_cm(y_true, y_pred, labels=None, method=None, subscript=None):
    if labels is None:
        if method == "clf":
            labels = ["Efferent+Afferent Motor", "Sensory", "Efferent motor",
                      "Dynamic", "Acoustic-mnestic", "Dysarthria", "Afferent motor"]
        elif method == "reg":
            labels = ["Mild", "Mild-moderate", "Moderate", "Moderate-severe", "Severe", "Very severe"]

    cm = confusion_matrix(y_true, y_pred)
    return plot_confusion_matrix(cm, target_names=labels, subscript=subscript)


class ObjectView(dict):
    def __init__(self, *args, **kwargs):
        super(ObjectView, self).__init__(**kwargs)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    self[key] = val
            else:
                raise TypeError()
        for key, val in kwargs.items():
            self[key] = val

    def __setattr__(self, key, value):
        if not hasattr(ObjectView, key):
            self[key] = value
        else:
            raise

    def __setitem__(self, name, value):
        value = ObjectView(value) if isinstance(value, dict) else value
        super(ObjectView, self).__setitem__(name, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, name):
        if name not in self:
            self[name] = {}
        return super(ObjectView, self).__getitem__(name)

    def __delattr__(self, name):
        del self[name]


def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def get_tqdm():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return tqdm_notebook
        if 'terminal' in ipy_str:
            return tqdm
    except:
        if sys.stderr.isatty():
            return tqdm
        else:
            def tqdm_dummy(iterable, **kwargs):
                return iterable
            return tqdm_dummy
