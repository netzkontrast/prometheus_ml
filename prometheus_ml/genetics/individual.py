import numpy as np
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.feature_selection.base import SelectorMixin
import joblib
import os


class Individual(BaseEstimator, MetaEstimatorMixin, SelectorMixin):

    def __init__(self, estimator=None, cv=None, scoring=None,
                 n_jobs=0, filename=None):

        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.scores_cache = {}
        self.filename = filename
        '''
        self.cache = {
            'holdout_score': float(estimator.best_score_['oof']['auc']),
            'holdout_prediction_folds': oof_holdout,
            'estimator_scores': estimator.best_score_,
            'oof_test_folds': oof_test_skf,
            'oof_train': oof_train,
            'oof_test_mean': oof_test,
            'estimator_params': estimator.get_params(),
            'estimator_feature_importance': estimator.feature_importances_,
            'estimator_best_iteration': int(estimator.best_iteration_),
            'estimator_n_features_': estimator.n_features_,
            'original_n_features': X.shape[0],
            'cv_scores': scores,
            'cv_score': scores_mean,
            'cv_score_std': scores_std,
            'folds': fold,
            'individual': individual,
            'individual_hash': str(hash(tuple(individual))),
            'time': time.time()
        }
        '''

    @staticmethod
    def restore(filename):
        data_dict = joblib.load(filename)
        return data_dict


    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @staticmethod
    def get_storage_path(filename):
        return os.path.join(os.getcwd(), filename)

    def save(self):
        filename = self.get_storage_path('cache.z')
        joblib.dump(self.scores_cache, filename, compress=True)

    def fit(self, X, y):
        return self._fit(X, y)

    def _fit(self, X, y):
        return self

    def _get_support_mask(self):
        pass


    @staticmethod
    def rounded_std(value, decimals=6):
        std = np.std(value, axis=0)
        return [np.round(std[0], decimals=decimals), np.round(std[1], decimals=1)]

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))

