"""
cv_score.py
-----------
This module provides classes and methods for computing a cross validation score.
By: Sebastian D. Goodfellow, Ph.D.
"""

# Local imports
from kardioml import LABELS_COUNT
from kardioml.scoring.scoring_metrics import compute_beta_score, compute_auc


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
        return {'train': {'accuracy': self.train_accuracy,
                          'f_measure': self.train_f_measure,
                          'f_beta': self.train_f_beta,
                          'g_beta': self.train_g_beta,
                          'auroc': self.train_auroc,
                          'auprc': self.train_auprc},
                'test': {'accuracy': self.test_accuracy,
                         'f_measure': self.test_f_measure,
                         'f_beta': self.test_f_beta,
                         'g_beta': self.test_g_beta,
                         'auroc': self.test_auroc,
                         'auprc': self.test_auprc}}

    def _compute_score(self):
        """Compute the CV score."""
        """Train Scores"""
        # Compute beta scores
        self.train_accuracy, self.train_f_measure, self.train_f_beta, self.train_g_beta = compute_beta_score(
            labels=self.y_train.values.tolist(), output=self.y_train_pred.tolist(), beta=2,
            num_classes=LABELS_COUNT, check_errors=True)

        # Compute AUC
        self.train_auroc, self.train_auprc = compute_auc(labels=self.y_train.values,
                                                         probabilities=self.y_train_proba,
                                                         num_classes=LABELS_COUNT, check_errors=True)

        """Test Scores"""
        # Compute beta scores
        self.test_accuracy, self.test_f_measure, self.test_f_beta, self.test_g_beta = compute_beta_score(
            labels=self.y_test.values.tolist(), output=self.y_test_pred.tolist(), beta=2,
            num_classes=LABELS_COUNT, check_errors=True)

        # Compute AUC
        self.test_auroc, self.test_auprc = compute_auc(labels=self.y_test.values, probabilities=self.y_test_proba,
                                                       num_classes=LABELS_COUNT, check_errors=True)
