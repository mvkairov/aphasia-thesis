from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from common.metrics import multiclass_roc_auc
from models.ordinal import OrdinalClassifier

def cv_roc_auc(clf, params, X, y, cv, n_classes):
    scores = []
    for train_index, val_index in cv.split(X, y):
        clf_instance = OrdinalClassifier(clf, **params)
        clf_instance.fit(X[train_index], y[train_index])
        y_pred = clf_instance.predict(X[val_index])
        scores.append(multiclass_roc_auc(y[val_index], y_pred, n_classes))
    return sum(scores) / len(scores)


def objective_wrapper(model_id, X, y, cv, n_classes):
    classifiers = {
        "knn": KNeighborsClassifier,
        "mlp": MLPClassifier,
        "rf": RandomForestClassifier,
        "gb": GradientBoostingClassifier,
        "svm": SVC
    }

    def knn_obj(trial):
        params = dict(
            n_neighbors = trial.suggest_int("n_neighbors", 2, 16),
            weights = trial.suggest_categorical("weights", ["uniform", "distance"]),
            algorithm = trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree"]),
            leaf_size = trial.suggest_int("leaf_size", 2, 32),
            p = trial.suggest_float("p", 1, 5)
        )
        return cv_roc_auc(classifiers[model_id], params, X, y, cv, n_classes)
    
    def mlp_obj(trial):
        params = dict(
            hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 16, 1024),
            max_iter = trial.suggest_int("max_iter", 16, 1000),
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-3, log=True),
            activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"]),
            solver = trial.suggest_categorical("solver", ["sgd", "adam"])
        )
        return cv_roc_auc(classifiers[model_id], params, X, y, cv, n_classes)
    
    def rf_obj(trial):
        params = dict(
            n_estimators = trial.suggest_int("n_estimators", 16, 1024),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 16),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 16),
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            max_depth = trial.suggest_int("max_depth", 2, 64),
            n_jobs = trial.suggest_categorical("n_jobs", [-1])
        )
        return cv_roc_auc(classifiers[model_id], params, X, y, cv, n_classes)

    def gb_obj(trial):
        params = dict(
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 1, log=True),
            n_estimators = trial.suggest_int("n_estimators", 16, 1024),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 16),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 16),
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            max_depth = trial.suggest_int("max_depth", 2, 64),
        )
        return cv_roc_auc(classifiers[model_id], params, X, y, cv, n_classes)

    def svm_obj(trial):
        params = dict(
            C = trial.suggest_float("C", 2 ** -3, 2 ** 15, log=True),
            gamma = trial.suggest_float("gamma", 2 ** -15, 2 ** 3, log=True),
            probability = trial.suggest_categorical("probability", [True])
        )
        return cv_roc_auc(classifiers[model_id], params, X, y, cv, n_classes)
    
    objectives = {
        "knn": knn_obj,
        "mlp": mlp_obj,
        "rf": rf_obj,
        "gb": gb_obj,
        "svm": svm_obj
    }

    return objectives[model_id]
