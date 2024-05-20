from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import label_binarize
from sklearn.metrics import make_scorer
import numpy as np

from common.utils import reg_to_clf_target


def multiclass_roc_auc(y_true, y_pred, n_classes, average="weighted"):
    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        y_true = label_binarize(y_true, classes=list(range(n_classes)))
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        y_pred = label_binarize(y_pred, classes=list(range(n_classes)))
    return roc_auc_score(y_true, y_pred, average=average, multi_class="ovr")


def get_clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, average: str = "weighted") -> dict[str, float]:
    metrics = {
        # "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "roc_auc": multiclass_roc_auc(y_true, y_pred, n_classes, average)
    }
    return metrics


def get_reg_as_clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str, n_classes: int, average: str = "weighted"):
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    y_true = reg_to_clf_target(y_true, target, n_classes)
    y_pred = reg_to_clf_target(y_pred, target, n_classes)

    metrics.update({
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "roc_auc": multiclass_roc_auc(y_true, y_pred, n_classes, average),
    })
    return metrics


def get_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        # "mape": mean_absolute_percentage_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    return metrics


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def get_val_metrics(method, reg_as_clf, target, n_classes):
    def clf_metric_wrapper(y_true, y_pred):
        return get_clf_metrics(y_true, y_pred, n_classes)
    
    def reg_as_clf_metric_wrapper(y_true, y_pred):
        return get_reg_as_clf_metrics(y_true, y_pred, target, n_classes)

    if method == "clf":
        return clf_metric_wrapper
    elif reg_as_clf:
        return reg_as_clf_metric_wrapper
    else:
        return get_reg_metrics


def get_tune_metrics(method, n_classes):
    if method == "clf":
        return make_scorer(multiclass_roc_auc, n_classes=n_classes)
    else:
        return make_scorer(r2_score)
