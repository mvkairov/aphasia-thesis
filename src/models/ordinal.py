from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
import numpy as np


class OrdinalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf=None, *args, **kwargs):
        self.clf = clf
        self.clfs = {}
        self.unique_class = np.NaN
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                # clf = clone(self.clf)
                clf = self.clf(*self.args, **self.kwargs)
                clf.fit(X, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {i: self.clfs[i].predict_proba(X) for i in self.clfs}
        predicted = []
        k = len(self.unique_class) - 1
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[0][:,1])
            elif i < k:
                # Vi = Pr(y <= Vi) * Pr(y > Vi-1)
                 predicted.append((1 - clfs_predict[i][:,1]) * clfs_predict[i-1][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[k-1][:,1])
        return np.vstack(predicted).T

    def predict(self, X):
        return self.unique_class[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
