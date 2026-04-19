import pandas as pd
import jax
from jax import jit, random
import jax.numpy as jnp
from functools import partial
import time
import mlflow


class logistic:
    """
    Basic Model + Quasi Newton Methods
    """
    def __init__(self, regularization='l2', method_opt='classic_model'):
        self.regularization = regularization
        self.method_opt = method_opt
        self.error_gradient = 0.001
        self.key = random.PRNGKey(73)
        self.W = None

    @staticmethod
    @jit
    def logistic_exp(W: jnp, X: jnp) -> jnp:
        """
        Generate all the w^T@x values
        args:
            W is a k-1 x d + 1
            X is a d x N
        """
        z = jnp.clip(W@X, -500.0, 500.0)  # clip data to prevent infinite values
        return jnp.exp(z)

    @staticmethod
    @jit
    def logistic_sum(exTerms: jnp) -> jnp:
        """
        Generate all the w^T@x values
        args:
            W is a k-1 x d
            X is a d x N
        """
        temp = jnp.sum(exTerms, axis=0)
        n = temp.shape[0]
        return jnp.reshape(1.0+temp, (1, n))

    @staticmethod
    @jit
    def logit_matrix(Terms: jnp, sum_terms: jnp) -> jnp:
        """
        Generate matrix
        """
        divisor = 1/sum_terms
        n, _ = Terms.shape
        replicate = jnp.repeat(divisor, repeats=n, axis=0)
        logits = Terms*replicate
        return jnp.vstack([logits, divisor])

    @partial(jit, static_argnums=(0,))
    def model(self, W: jnp, X: jnp, Y_hot: jnp) -> jnp:
        """
        Logistic Model
        """
        W = jnp.reshape(W, self.sh)
        terms = self.logistic_exp(W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        matrix_safe = jnp.clip(matrix, 1e-15, 1.0)
        return jnp.sum(jnp.sum(jnp.log(matrix_safe) * Y_hot, axis=0), axis=0)

    @staticmethod
    def one_hot(Y: jnp):
        """
        One_hot matrix
        """
        numclasses = len(jnp.unique(Y))
        return jnp.transpose(jax.nn.one_hot(Y, num_classes=numclasses))

    def generate_w(self, k_classes: int, dim: int) -> jnp:
        """
        Use the random generator at Jax to generate a random generator to instanciate
        the augmented values
        """
        key = random.PRNGKey(0)
        keys = random.split(key, 1)
        return jnp.array(random.normal(keys[0], (k_classes, dim)))

    @staticmethod
    def augment_x(X: jnp) -> jnp:
        """
        Augmenting samples of a dim x N matrix
        """
        N = X.shape[1]
        return jnp.vstack([X, jnp.ones((1, N))])

    def fit(self, X: jnp, Y: jnp) -> None:
        """
        The fit process
        """
        nclasses = len(jnp.unique(Y))
        X = self.augment_x(X)
        dim = X.shape[0]
        W = self.generate_w(nclasses-1, dim)
        Y_hot = self.one_hot(Y)
        self.W = getattr(self, self.method_opt, lambda W, X, Y_hot: self.error())(W, X, Y_hot)

    @staticmethod
    def error() -> None:
        """
        Only Print Error
        """
        raise Exception("Opt Method does not exist")

    def classic_model(self, W: jnp, X: jnp, Y_hot: jnp, alpha: float = 1e-2, tol: float = 1e-3) -> jnp:
        """
        The naive version of the logistic regression
        """
        n, m = W.shape
        self.sh = (n, m)
        Grad = jax.grad(self.model, argnums=0)(jnp.ravel(W), X, Y_hot)
        loss = self.model(jnp.ravel(W), X, Y_hot)
        cnt = 0
        while True:
            Hessian = jax.hessian(self.model, argnums=0)(jnp.ravel(W), X, Y_hot)

            # prevents singular
            mid_hessian = jnp.eye(Hessian.shape[0])
            Hessian_safe = Hessian + mid_hessian * 1e-5

            step = jnp.linalg.solve(Hessian_safe, Grad)
            W = W - alpha*jnp.reshape(step, self.sh)
            Grad = jax.grad(self.model, argnums=0)(jnp.ravel(W), X, Y_hot)
            old_loss = loss
            loss = self.model(jnp.ravel(W), X, Y_hot)

            if cnt % 30 == 0:
                time.sleep(0.1)
                mlflow.log_metric("loss", float(loss), step=cnt)
            if jnp.abs(old_loss - loss) < tol:
                break
            cnt += 1
            # emergency if it tends to infinite
            if cnt > 3000:
                print("Reached iteration limit")
                break
        return W

    def estimate_prob(self, X: jnp) -> jnp:
        """
        Estimate Probability
        """
        X = self.augment_x(X)
        terms = self.logistic_exp(self.W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        return matrix

    def estimate(self, X: jnp) -> jnp:
        """
        Estimation
        """
        X = self.augment_x(X)
        terms = self.logistic_exp(self.W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        return jnp.argmax(matrix, axis=0)

    def precision(self, y, y_hat):
        """
        Precision
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FP)
        """
        TP = sum(y_hat == y)
        FP = sum(y_hat != y)
        return (TP/(TP+FP)).tolist()

    def calculate_metrics(self, y, y_hat):
        """
        Calculate Precision, Recall and F1-Score for binary clasification
        """
        # True Positives
        TP = jnp.sum((y_hat == 1) & (y == 1))

        # False positives
        FP = jnp.sum((y_hat == 1) & (y == 0))

        # False Negative
        FN = jnp.sum((y_hat == 0) & (y == 1))
        TN = jnp.sum((y_hat == 0) & (y == 0))

        # confusion matrix
        cm = jnp.array([
            [TN, FP],
            [FN, TP]
        ])

        # jnp.maximum avoid zero division
        precision = TP / jnp.maximum(TP + FP, 1e-9)
        recall = TP / jnp.maximum(TP + FN, 1e-9)
        accuracy = (TP + TN) / jnp.maximum(TP + TN + FP + FN, 1e-9)
        f1_score = 2 * (precision * recall) / jnp.maximum(precision + recall, 1e-9)
        print(f'TP: {TP}')
        print(f'TN: {TN}')
        print(f'FP: {FP}')
        print(f'FN: {FN}')

        return precision.tolist(), recall.tolist(), accuracy.tolist(), f1_score.tolist(), cm
