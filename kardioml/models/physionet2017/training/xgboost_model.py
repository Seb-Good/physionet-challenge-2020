"""
xgboost_model.py
----------------
This module provides classes and methods for building a compositional data model.
By: Sebastian D. Goodfellow, Ph.D.
"""

# 3rd party imports
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization


class Model(object):

    def __init__(self, dataset):

        # Set parameters
        self.dataset = dataset

        # Set attributes
        self.model = None
        self.optimizer = None
        self.params = None
        self.cv_scores = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.train_mean_squared_error = None
        self.train_root_mean_squared_error = None
        self.train_mean_absolute_error = None
        self.train_r2_score = None
        self.test_mean_squared_error = None
        self.test_root_mean_squared_error = None
        self.test_mean_absolute_error = None
        self.test_r2_score = None

    def tune_hyper_parameters(self, param_bounds, n_iter=10):
        """Run hyper-parameter optimization."""
        # Initialize optimizer
        self.optimizer = BayesianOptimization(f=self.cross_validate,
                                              pbounds=param_bounds,
                                              random_state=0)

        # Run optimization
        self.optimizer.maximize(init_points=5, n_iter=n_iter, acq='ucb', kappa=3, alpha=1e-10)

        # Get final hyper-parameters
        self.params = self.optimizer.max['params']

        # Compute final cv scores
        self.cv_scores = self._compute_final_cv_scores()

        # Train final model
        self.train_final_model(params=self.params)

    def cross_validate(self, learning_rate, n_estimators, max_depth, subsample, colsample,
                       gamma, min_child_weight, max_delta_step):
        """Run cross validation."""
        # Force data type
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)

        # Set parameters
        params = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 'max_depth': max_depth,
                  'subsample': subsample, 'colsample': colsample, 'gamma': gamma,
                  'min_child_weight': min_child_weight, 'max_delta_step': max_delta_step}

        # Initialize model
        model = XGBRegressor(learning_rate=learning_rate,
                             n_estimators=n_estimators,
                             max_depth=max_depth,
                             subsample=subsample,
                             colsample=colsample,
                             gamma=gamma,
                             min_child_weight=min_child_weight,
                             max_delta_step=max_delta_step)

        # Compute CV scores
        cv_scores = self.compute_cv_scores(model=model, params=params)

        # Compute score for optimization
        return -np.mean([cv_score['cv_score'].mean_absolute_error for cv_index, cv_score in cv_scores.items()])

    def compute_cv_scores(self, model, params):
        """Compute cross validation metrics."""
        # Dictionary for cv_scores
        cv_scores = dict()

        # Loop through folds
        for cv_index in self.dataset.cv_folds.keys():

            # Compute CV score
            cv_score = CVScore(model=model, cv_index=cv_index,
                               train_index=self.dataset.cv_folds[cv_index]['train_index'],
                               test_index=self.dataset.cv_folds[cv_index]['test_index'],
                               x=self.dataset.x_train,
                               y=self.dataset.y_train)

            # Collect cv score
            cv_scores[cv_index] = {'cv_score': cv_score, 'params': params}

        return cv_scores

    def _compute_final_cv_scores(self):
        """Compute final cv scores after optimization."""
        # Initialize model
        model = XGBRegressor(learning_rate=self.params['learning_rate'],
                             n_estimators=int(self.params['n_estimators']),
                             max_depth=int(self.params['max_depth']),
                             subsample=self.params['subsample'],
                             colsample=self.params['colsample'],
                             gamma=self.params['gamma'],
                             min_child_weight=self.params['min_child_weight'],
                             max_delta_step=self.params['max_delta_step'])

        # Compute CV scores
        cv_scores = self.compute_cv_scores(model=model, params=self.params)

        return cv_scores

    def train_final_model(self, params):
        """Train final model after optimization."""
        # Initialize model
        self.model = XGBRegressor(learning_rate=params['learning_rate'],
                                  n_estimators=int(params['n_estimators']),
                                  max_depth=int(params['max_depth']),
                                  subsample=params['subsample'],
                                  colsample=params['colsample'],
                                  gamma=params['gamma'],
                                  min_child_weight=params['min_child_weight'],
                                  max_delta_step=params['max_delta_step'])

        # Train model
        self.model.fit(X=self.dataset.x_train, y=self.dataset.y_train)

        # Compute final train-test metrics
        self._compute_final_train_test_scores()

    def _compute_final_train_test_scores(self):
        """Compute final cv scores after optimization."""
        # Compute train metrics
        self.y_train_pred = pd.Series(data=self.model.predict(self.dataset.x_train), index=self.dataset.x_train.index)
        self.train_mean_squared_error = mean_squared_error(y_pred=self.y_train_pred, y_true=self.dataset.y_train)
        self.train_root_mean_squared_error = np.sqrt(self.train_mean_squared_error)
        self.train_mean_absolute_error = mean_absolute_error(y_pred=self.y_train_pred, y_true=self.dataset.y_train)
        self.train_r2_score = r2_score(y_pred=self.y_train_pred, y_true=self.dataset.y_train)

        # Compute test metrics
        self.y_test_pred = pd.Series(data=self.model.predict(self.dataset.x_test), index=self.dataset.x_test.index)
        self.test_mean_squared_error = mean_squared_error(y_pred=self.y_test_pred, y_true=self.dataset.y_test)
        self.test_root_mean_squared_error = np.sqrt(self.test_mean_squared_error)
        self.test_mean_absolute_error = mean_absolute_error(y_pred=self.y_test_pred, y_true=self.dataset.y_test)
        self.test_r2_score = r2_score(y_pred=self.y_test_pred, y_true=self.dataset.y_test)

    def get_final_train_test_scores(self):
        """Return a dictionary of final scores."""
        return {'train': {'mean_squared_error': self.train_mean_squared_error ,
                          'root_mean_squared_error': self.train_root_mean_squared_error,
                          'mean_absolute_error ': self.train_mean_absolute_error,
                          'r2_score': self.train_r2_score},
                'test': {'mean_squared_error': self.test_mean_squared_error,
                         'root_mean_squared_error': self.test_root_mean_squared_error,
                         'mean_absolute_error ': self.test_mean_absolute_error,
                         'r2_score': self.test_r2_score}}

    def get_final_cv_scores(self):
        """Return a dictionary of final scores."""
        return {cv_index: self.cv_scores[cv_index]['cv_score'].get_cv_score()
                for cv_index in range(len(self.cv_scores))}

    def save(self, path):
        """Pickle data model."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
