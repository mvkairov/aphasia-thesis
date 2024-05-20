from common.utils import make_name, make_msg, save_to_pkl, get_cv, ObjectView
from common.metrics import get_val_metrics, get_tune_metrics
from train.tab import search_hp, cross_validate
from data.tab import get_tabular_data
from tune.spaces import tab_spaces

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

from omegaconf import OmegaConf
import numpy as np
import hydra

np.int = np.int32
models = {
    "clf": {
        "knn": KNeighborsClassifier,
        "mlp": MLPClassifier,
        "rf": RandomForestClassifier,
        "gb": GradientBoostingClassifier,
        "svm": SVC,
        "cb": CatBoostClassifier,
        "xgb": XGBClassifier
    },
    "reg": {
        "knn": KNeighborsRegressor,
        "mlp": MLPRegressor,
        "rf": RandomForestRegressor,
        "gb": GradientBoostingRegressor,
        "svm": SVR,
        "cb": CatBoostRegressor,
        "xgb": XGBRegressor
    }
}


@hydra.main(version_base=None, config_path="../config", config_name="tab")
def my_app(cfg):
    cfg = ObjectView(OmegaConf.to_container(cfg))
    run_name = make_name(cfg)

    X, y, _, n_classes = get_tabular_data(**cfg.ds)
    model = models[cfg.ds.method][cfg.model_id]
    train_cv = get_cv(cfg.ds.method, cfg.n_cv_splits, reg_as_clf=cfg.reg_as_clf, shuffle=True, target=cfg.ds.target, n_classes=n_classes)
    tune_cv = get_cv(cfg.ds.method, cfg.n_tune_splits, reg_as_clf=cfg.reg_as_clf, shuffle=True, target=cfg.ds.target, n_classes=n_classes)
    objective = get_tune_metrics(cfg.ds.method, n_classes)
    metrics = get_val_metrics(cfg.ds.method, cfg.reg_as_clf, cfg.ds.target, n_classes)

    best_params = search_hp(X, y, model, tab_spaces[cfg.model_id], objective, tune_cv, cfg.max_trials, cfg.n_jobs)
    scores, predictions = cross_validate(X, y, model, best_params, train_cv, metrics)

    print(make_msg(run_name, best_params, scores))
    save_to_pkl(predictions, run_name, "results/predictions")
    save_to_pkl(best_params, run_name, "results/params")
    save_to_pkl(scores, run_name, "results/scores")


if __name__ == "__main__":
    my_app()
