"""
xgboost_model.py
----------------
This module provides classes and methods for building a compositional data model.
By: Sebastian D. Goodfellow, Ph.D.
"""

# 3rd party imports
import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

# Local imports
from kardioml import WORKING_PATH, FILTER_BAND_LIMITS
from kardioml.models.physionet2017.training.cv_score import CVScore
from kardioml.models.physionet2017.features.feature_extractor import Features


class Model(object):
    def __init__(self, features, labels, cv_folds, stratifier):

        # Set parameters
        self.features = features
        self.labels = labels
        self.cv_folds = cv_folds
        self.stratifier = stratifier

        # Set attributes
        self.model = None
        self.optimizer = None
        self.params = None
        self.stratified_kfolds = None
        self.cv_scores = None

    def tune_hyper_parameters(self, param_bounds, n_iter=10):
        """Run hyper-parameter optimization."""
        # Initialize optimizer
        self.optimizer = BayesianOptimization(f=self.cross_validate, pbounds=param_bounds, random_state=0)

        # Run optimization
        self.optimizer.maximize(init_points=5, n_iter=n_iter, acq='ucb', kappa=3, alpha=1e-10)

        # Get final hyper-parameters
        self.params = self.optimizer.max['params']

        # Compute final cv scores
        self.cv_scores = self._compute_final_cv_scores()

        # Train final model
        self.train_final_model(params=self.params)

    def cross_validate(
        self,
        learning_rate,
        n_estimators,
        max_depth,
        subsample,
        colsample_bytree,
        gamma,
        min_child_weight,
        max_delta_step,
    ):
        """Run cross validation."""
        # Force data type
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)

        # Set parameters
        params = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'max_delta_step': max_delta_step,
        }

        # Initialize model
        model = OneVsRestClassifier(
            XGBClassifier(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                min_child_weight=min_child_weight,
                max_delta_step=max_delta_step,
                n_jobs=-1,
            )
        )

        # Compute CV scores
        cv_scores = self.compute_cv_scores(model=model, params=params)

        # Compute score for optimization
        return np.mean([cv_score['cv_score'].test_accuracy for cv_index, cv_score in cv_scores.items()])

    def compute_cv_scores(self, model, params):
        """Compute cross validation metrics."""
        # Dictionary for cv_scores
        cv_scores = dict()

        # Get stratified folds
        self.stratified_kfolds = StratifiedKFold(n_splits=self.cv_folds, random_state=0, shuffle=True)

        # Loop through folds
        for cv_index, (train_index, test_index) in enumerate(
            self.stratified_kfolds.split(X=self.features, y=self.stratifier)
        ):

            # Compute CV score
            cv_score = CVScore(
                model=model,
                cv_index=cv_index,
                train_index=train_index,
                test_index=test_index,
                x=self.features,
                y=self.labels,
            )

            # Collect cv score
            cv_scores[cv_index] = {'cv_score': cv_score, 'params': params}

        return cv_scores

    def _compute_final_cv_scores(self):
        """Compute final cv scores after optimization."""
        # Initialize model
        model = OneVsRestClassifier(
            XGBClassifier(
                learning_rate=self.params['learning_rate'],
                n_estimators=int(self.params['n_estimators']),
                max_depth=int(self.params['max_depth']),
                subsample=self.params['subsample'],
                colsample_bytree=self.params['colsample_bytree'],
                gamma=self.params['gamma'],
                min_child_weight=self.params['min_child_weight'],
                max_delta_step=self.params['max_delta_step'],
                n_jobs=-1,
            )
        )

        # Compute CV scores
        cv_scores = self.compute_cv_scores(model=model, params=self.params)

        return cv_scores

    def train_final_model(self, params):
        """Train final model after optimization."""
        # Initialize model
        self.model = OneVsRestClassifier(
            XGBClassifier(
                learning_rate=params['learning_rate'],
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                min_child_weight=params['min_child_weight'],
                max_delta_step=params['max_delta_step'],
                n_jobs=-1,
            )
        )

        # Train model
        self.model.fit(X=self.features, y=self.labels)

    def get_final_cv_scores(self):
        """Return a dictionary of final scores."""
        return {
            cv_index: self.cv_scores[cv_index]['cv_score'].get_cv_score()
            for cv_index in range(len(self.cv_scores))
        }

    def save(self):
        """Pickle data model."""
        # Create directory for formatted data
        os.makedirs(os.path.join(WORKING_PATH, 'models', 'physionet2017'), exist_ok=True)

        with open(os.path.join(WORKING_PATH, 'models', 'physionet2017', 'physionet2017.model'), 'wb') as f:
            pickle.dump(self, f)

    def challenge_prediction(self, data, header_data, lead):
        """Get final predictions for competition inference."""
        # Get features
        features = Features(filename=None, waveform_data=data, header_data=header_data, lead=lead)
        features.extract_features(
            feature_groups=['full_waveform_features', 'rri_features', 'template_features'],
            filter_bandwidth=FILTER_BAND_LIMITS,
        )

        # Get prediction output
        predictions = self.model.predict(features.get_features()).flatten()
        probabilities = self.model.predict_proba(features.get_features()).flatten()

        return predictions, probabilities
