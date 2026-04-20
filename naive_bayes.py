import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats


class naive_bayes():
    def __init__(self, pi, mu, sigma, X):
        self.X = X
        self.pi_l = pi[0]
        self.pi_k = pi[1]
        self.mu_l = mu[0]
        self.mu_k = mu[1]
        self.sigma_1 = jnp.diag(jnp.diag(sigma[0])) # extract diagonal and complete with zeros
        self.sigma_2 = jnp.diag(jnp.diag(sigma[1])) # extract diagonal and complete with zeros

    def model(self):
        """
        classification model
        """
        log_pi_l = jnp.log(self.pi_l)
        log_pi_k = jnp.log(self.pi_k)

        # discriminants
        # g_k(x) = ln(pi_k) + ln(N(x | mu_k, sigma_diag_k))
        g_1 = log_pi_l + jstats.multivariate_normal.logpdf(self.X, self.mu_l, self.sigma_1)
        g_2 = log_pi_k + jstats.multivariate_normal.logpdf(self.X, self.mu_k, self.sigma_2)

        # decission rule. If P(w1|x) > P(w2|x) x is classified to w1, If P(w1|x) < P(w2|x) x is classified to w2
        # we get the max value for each
        G = jnp.column_stack([g_1, g_2])
        self.y_pred = jnp.argmax(G, axis=1)

        return self
    
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

        # jnp.maximum avoid zero division
        precision = TP / jnp.maximum(TP + FP, 1e-9)
        recall = TP / jnp.maximum(TP + FN, 1e-9)
        accuracy = (TP + TN) / jnp.maximum(TP + TN + FP + FN, 1e-9)
        f1_score = 2 * (precision * recall) / jnp.maximum(precision + recall, 1e-9)

        return precision.tolist(), recall.tolist(), accuracy.tolist(), f1_score.tolist()