from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
import optuna

from common.utils import get_dict_of_lists, sklearn_best_params
from tune.trials import objective_wrapper


def search_hp(X, y, model, params, objective, cv, max_trials, n_jobs):
    pipeline = BayesSearchCV(
        Pipeline([
            ("model", model()),
        ]),
        dict(zip([f"model__{param}" for param in params], list(params.values()))),
        scoring=objective, cv=cv, n_iter=max_trials, n_points=n_jobs, n_jobs=n_jobs
    )
    pipeline.fit(X, y)
    search_results = pipeline.cv_results_
    best_params = sklearn_best_params(search_results)
    return best_params


def search_ord_hp(model_id, X, y, cv, n_classes, max_trials, n_jobs):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_wrapper(model_id, X, y, cv, n_classes), n_trials=max_trials, n_jobs=n_jobs)
    return study.best_params


def cross_validate(X, y, model, params, cv, metrics):
    scores, predictions = [], []
    for train_index, val_index in cv.split(X, y):
        model_instance = model(**params)
        model_instance.fit(X[train_index], y[train_index])
        y_pred = model_instance.predict(X[val_index])
        scores.append(metrics(y[val_index], y_pred))
        predictions.append((y[val_index].tolist(), y_pred.tolist()))
    scores = get_dict_of_lists(scores, return_mean_std=True)
    return scores, predictions
