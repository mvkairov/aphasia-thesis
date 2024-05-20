from skopt.space import Real, Categorical, Integer

tab_spaces = {
    "mlp": {
        "hidden_layer_sizes": Integer(16, 1024),
        "max_iter": Integer(16, 5000),
        "learning_rate_init": Real(1e-5, 1e-3, prior="log-uniform"),
        "activation": Categorical(["identity", "logistic", "tanh", "relu"]),
        "solver": Categorical(["sgd", "adam"]),
    },
    "rf": {
        "n_estimators": Integer(16, 8192),
        "min_samples_split": Integer(2, 16),
        "min_samples_leaf": Integer(1, 16),
        "max_features": ["sqrt", "log2"],
        "max_depth": Integer(2, 64),
        "n_jobs": [-1]
    },
    "gb": {
        "learning_rate": Real(10 ** (-3), 1),
        "n_estimators": Integer(16, 8192),
        "min_samples_split": Integer(2, 16),
        "min_samples_leaf": Integer(1, 16),
        "max_features": Categorical(["sqrt", "log2"]),
        "max_depth": Integer(2, 64),
    },
    "knn": {
        "n_neighbors": Integer(2, 16),
        "weights": Categorical(["uniform", "distance"]),
        "algorithm": Categorical(["ball_tree", "kd_tree"]),
        "leaf_size": Integer(2, 32),
        "p": Real(1, 5)
    },
    "svm": {
        "C": Real(2 ** -3, 2 ** 15, prior="log-uniform"),
        "gamma": Real(2 ** -15, 2 ** 3, prior="log-uniform")
    },
    "cb": {
        "iterations": Integer(2, 5000),
        "depth": Integer(2, 16),
        "learning_rate": Real(1e-3, 1, prior="log-uniform"),
        "verbose": [False]
    },
    "xgb": {
        "max_depth": Integer(2, 64),
        "n_estimators": Integer(10, 8192),
        "verbosity": [0]
    }
}
