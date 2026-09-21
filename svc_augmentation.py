"""SVC с перестановкой половин признаков после предобработки."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SwapAugmentedSVC(ClassifierMixin, BaseEstimator):
    """При fit добавляет строки с переставленными половинами признаков.

    Первые ``n_pair_features`` столбцов образуют две половины пары. Остальные
    столбцы (агрегаты) сохраняются. При predict данные не расширяются.
    """

    def __init__(self, estimator=None, n_pair_features=512):
        self.estimator = estimator
        self.n_pair_features = n_pair_features

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.n_pair_features <= 0 or self.n_pair_features % 2:
            raise ValueError('n_pair_features должно быть положительным чётным числом')
        if self.n_pair_features > X.shape[1]:
            raise ValueError('n_pair_features больше числа столбцов X')

        half = self.n_pair_features // 2
        swapped = np.concatenate(
            [X[:, half:self.n_pair_features], X[:, :half], X[:, self.n_pair_features:]],
            axis=1,
        )
        X_aug = np.concatenate([X, swapped], axis=0)
        y_aug = np.concatenate([y, y])

        self.estimator_ = clone(self.estimator) if self.estimator is not None else SVC()
        self.estimator_.fit(X_aug, y_aug)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.decision_function(check_array(X))

    def predict(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(check_array(X))
