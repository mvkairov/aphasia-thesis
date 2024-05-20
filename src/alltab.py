from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from omegaconf import OmegaConf
from itertools import product
import numpy as np
import hydra

from common.utils import save_to_pkl, get_cv, ObjectView, make_name
from train.tab import search_hp, cross_validate, search_ord_hp
from common.metrics import get_val_metrics, get_tune_metrics
from models.ordinal import OrdinalClassifier
from data.tab import get_tabular_data
from tune.spaces import tab_spaces

np.int = np.int32
models = {
    "clf": {
        "knn": KNeighborsClassifier,
        "mlp": MLPClassifier,
        "rf": RandomForestClassifier,
        "gb": GradientBoostingClassifier,
        "svm": SVC
    },
    "reg": {
        "knn": KNeighborsRegressor,
        "mlp": MLPRegressor,
        "rf": RandomForestRegressor,
        "gb": GradientBoostingRegressor,
        "svm": SVR
    }
}

matters = ["grey", "white", "both"]
demos = [False, True]
model_ids = ["knn", "mlp", "rf", "gb", "svm"]


@hydra.main(version_base=None, config_path="../config", config_name="tab")
def my_app(cfg):
    cfg = ObjectView(OmegaConf.to_container(cfg))
    res = {}
    for model_id, matter, demo in product(model_ids, matters, demos):
        cfg.ds.matter = matter
        cfg.ds.demo = demo
        cfg.model_id = model_id

        X, y, _, n_classes = get_tabular_data(**cfg.ds)
        model = models[cfg.ds.method][cfg.model_id]
        train_cv = get_cv(cfg.ds.method, cfg.n_cv_splits, reg_as_clf=cfg.reg_as_clf, shuffle=True, target=cfg.ds.target, n_classes=n_classes)
        tune_cv = get_cv(cfg.ds.method, cfg.n_tune_splits, reg_as_clf=cfg.reg_as_clf, shuffle=True, target=cfg.ds.target, n_classes=n_classes)
        objective = get_tune_metrics(cfg.ds.method, n_classes)
        metrics = get_val_metrics(cfg.ds.method, cfg.reg_as_clf, cfg.ds.target, n_classes)

        if cfg.ordinal:
            best_params = search_ord_hp(cfg.model_id, X, y, tune_cv, n_classes, cfg.max_trials, cfg.n_jobs)
            best_params["clf"] = models[cfg.ds.method][cfg.model_id]
            model = OrdinalClassifier
        else:
            best_params = search_hp(X, y, model, tab_spaces[cfg.model_id], objective, tune_cv, cfg.max_trials, cfg.n_jobs)
        scores, predictions = cross_validate(X, y, model, best_params, train_cv, metrics)

        res[make_name(cfg)] = {
            "params": best_params,
            "scores": scores,
            "predictions": predictions
        }
    save_to_pkl(res, cfg.name, "results")


if __name__ == "__main__":
    my_app()
