import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, random


class linear:
    def __init__(self):
        self.beta = None

    @staticmethod
    @jit
    def augmented_X(X: jnp) -> jnp:
        """
        Add beta_0 to the vector to add the property of no singular to the transpose matrix

        :param self: Description
        """
        N = X.shape[1]
        return jnp.vstack([X, jnp.ones((1, N))])

    def fit(self, X: jnp, y: jnp) -> None:
        """
        Adjust model to normal ecuation. X should be (d x N)
        """
        X_aug = self.augmented_X(X)
        XtX = jnp.dot(X_aug, X_aug.T)
        Xty = jnp.dot(X_aug, y)
        self.beta = jnp.linalg.solve(XtX, Xty)

    def estimate_continuous(self, X: jnp) -> jnp:
        """
        return raw predictions
        """
        X_aug = self.augmented_X(X)
        return self.beta.T@X_aug

    def estimate(self, X: jnp, threshold: float = 0.5) -> jnp:
        """
        Use regression as classificator, where 1 is an exoplanet and 0 isn't
        """
        y_hat_continuous = self.estimate_continuous(X)
        return jnp.where(y_hat_continuous >= threshold, 1, 0)

    def calculate_metrics(self, y: jnp, y_hat: jnp):
        """
        Calculate Precision, Recall y F1-Score for binary classification
        """
        TP = jnp.sum((y_hat == 1) & (y == 1))
        FP = jnp.sum((y_hat == 1) & (y == 0))
        FN = jnp.sum((y_hat == 0) & (y == 1))

        # jnp.maximum avoid zero division
        precision = TP / jnp.maximum(TP + FP, 1e-9)
        recall = TP / jnp.maximum(TP + FN, 1e-9)
        f1_score = 2 * (precision * recall) / jnp.maximum(precision + recall, 1e-9)

        return precision.tolist(), recall.tolist(), f1_score.tolist()