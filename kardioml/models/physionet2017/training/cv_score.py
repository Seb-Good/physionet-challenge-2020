"""
cv_score.py
-----------
This module provides classes and methods for computing a cross validation score.
By: Sebastian D. Goodfellow, Ph.D.
"""

# Local imports
from kardioml import WEIGHTS_PATH
from kardioml.scoring.scoring_metrics import (
    load_weights,
    compute_challenge_metric,
    compute_auc,
    compute_beta_measures,
    compute_f_measure,
    compute_accuracy,
)


class CVScore(object):
    def __init__(self, model, cv_index, train_index, test_index, x, y):

        # Set parameters
        self.model = model
        self.cv_index = cv_index
        self.train_index = train_index
        self.test_index = test_index
        self.x = x
        self.y = y

        # Set attributes
        self.x_train = self.x.loc[self.train_index, :]
        self.y_train = self.y.loc[self.train_index, :]
        self.x_test = self.x.loc[self.test_index, :]
        self.y_test = self.y.loc[self.test_index, :]
        self.train_accuracy = None
        self.train_f_measure = None
        self.train_f_beta = None
        self.train_g_beta = None
        self.train_auroc = None
        self.train_auprc = None
        self.test_accuracy = None
        self.test_f_measure = None
        self.test_f_beta = None
        self.test_g_beta = None
        self.test_auroc = None
        self.test_auprc = None

        # Train model
        self.model.fit(X=self.x_train, y=self.y_train)

        # Get train prediction
        self.y_train_pred = self.model.predict(self.x_train)
        self.y_train_proba = self.model.predict_proba(self.x_train)

        # Get test prediction
        self.y_test_pred = self.model.predict(self.x_test)
        self.y_test_proba = self.model.predict_proba(self.x_test)

        # Compute CV score
        self._compute_score()

    def get_cv_score(self):
        """Return a dictionary of final scores."""
        return {
            'train': {
                'accuracy': self.train_accuracy,
                'macro_f_measure': self.train_macro_f_measure,
                'macro_f_beta_measure': self.train_macro_f_beta_measure,
                'macro_g_beta_measure': self.train_macro_g_beta_measure,
                'macro_auroc': self.train_macro_auroc,
                'macro_auprc': self.train_macro_auprc,
                'challenge_metric': self.train_challenge_metric,
            },
            'test': {
                'accuracy': self.test_accuracy,
                'macro_f_measure': self.test_macro_f_measure,
                'macro_f_beta_measure': self.test_macro_f_beta_measure,
                'macro_g_beta_measure': self.test_macro_g_beta_measure,
                'macro_auroc': self.test_macro_auroc,
                'macro_auprc': self.test_macro_auprc,
                'challenge_metric': self.test_challenge_metric,
            },
        }

    def _compute_score(self):
        """Compute the CV score."""
        """Train Scores"""
        # Compute accuracy
        self.train_accuracy = compute_accuracy(labels=self.y_train.values, outputs=self.y_train_pred)

        # Compute F-measures
        self.train_macro_f_measure = compute_f_measure(labels=self.y_train.values, outputs=self.y_train_pred)

        # Compute Beta-measures
        self.train_macro_f_beta_measure, self.train_macro_g_beta_measure = compute_beta_measures(
            labels=self.y_train.values, outputs=self.y_train_pred, beta=2
        )

        # Compute AUC
        self.train_macro_auroc, self.train_macro_auprc = compute_auc(
            labels=self.y_train.values, outputs=self.y_train_pred
        )

        # Compute challenge metric
        weights = load_weights(weight_file=WEIGHTS_PATH, classes=self.y.columns.tolist())
        self.train_challenge_metric = compute_challenge_metric(
            weights=weights,
            labels=self.y_train.values,
            outputs=self.y_train_pred,
            classes=self.y.columns.tolist(),
            normal_class='426783006',
        )

        """Test Scores"""
        # Compute accuracy
        self.test_accuracy = compute_accuracy(labels=self.y_test.values, outputs=self.y_test_pred)

        # Compute F-measures
        self.test_macro_f_measure = compute_f_measure(labels=self.y_test.values, outputs=self.y_test_pred)

        # Compute Beta-measures
        self.test_macro_f_beta_measure, self.test_macro_g_beta_measure = compute_beta_measures(
            labels=self.y_test.values, outputs=self.y_test_pred, beta=2
        )

        # Compute AUC
        self.test_macro_auroc, self.test_macro_auprc = compute_auc(
            labels=self.y_test.values, outputs=self.y_test_pred
        )

        # Compute challenge metric
        weights = load_weights(weight_file=WEIGHTS_PATH, classes=self.y.columns.tolist())
        self.test_challenge_metric = compute_challenge_metric(
            weights=weights,
            labels=self.y_test.values,
            outputs=self.y_test_pred,
            classes=self.y.columns.tolist(),
            normal_class='426783006',
        )
