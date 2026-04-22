import jax
from jax import jit, random
import jax.numpy as jnp
from functools import partial
import jax.scipy.stats as jstats


class em_algorithm:
    """
    steps 2 to 4 for two-components gaussian mixture
    """
    def __init__(self, seed=73):
        self.key = random.PRNGKey(seed)
        self.pi = None
        self.mu = None
        self.sigma = None

    @partial(jit, static_argnums=(0,))
    def expectation_step(self, X, pi, mu, sigma):
        """
        Calculate responsibilities (step 2)
        """
        log_pdf1 = jstats.multivariate_normal.logpdf(X, mu[0], sigma[0])
        log_pdf2 = jstats.multivariate_normal.logpdf(X, mu[1], sigma[1])

        log_P = jnp.column_stack([log_pdf1, log_pdf2])

        log_P_weighted = log_P + jnp.log(pi)  # numerator on the equation

        # denominator
        # logsumexp cambia los datos para que no exista un underflow de datos, ademas de trabajar
        # con la loglikelihood
        log_P_evidence = jax.scipy.special.logsumexp(log_P_weighted, axis=1, keepdims=True)

        gamma = log_P_weighted - log_P_evidence
        gamma = jnp.exp(gamma)
        log_likelihood = jnp.sum(log_P_evidence)

        return gamma, log_likelihood
    
    @partial(jit, static_argnums=(0,))
    def maximization_step(self, X, gamma):
        """
        Compute the weighted means and variances (step 3)
        """
        N = X.shape[0]
        sum_gamma = jnp.sum(gamma, axis=0)
        new_pi = sum_gamma / N
        # sum_gamma is [2,] if we divide directly, it's going to crash, so we add a second column
        mu_hat = (gamma.T @ X) / sum_gamma[:,None]

        epsilon = 1e-6
        cluster1_to_X = X - mu_hat[0]
        sigma_1= (cluster1_to_X.T @ (cluster1_to_X * gamma[:, 0:1])) / sum_gamma[0]

        cluster2_to_X = X - mu_hat[1]
        sigma_2 = (cluster2_to_X.T @ (cluster2_to_X * gamma[:, 1:2])) / sum_gamma[1]
        sigma_2 = sigma_2 + jnp.eye(sigma_2.shape[0]) * epsilon  # for regularization

        sigma = jnp.stack([sigma_1, sigma_2])

        return new_pi, mu_hat, sigma

    def fit_em(self, X:jnp.ndarray, iterations: int = 100, tolerance: float=1e-4, n_init: int = 7):
        """
        iterate model until convergence (step 4 of algorithm)
        """
        N, d = X.shape
        best_pi = None
        best_mu = None
        best_sigma = None
        best_log_likelihood = -jnp.inf

        # diferent intialization to get the best result
        for em_try in range(n_init):
            # pi: we initialize pi at 0.5, same for both clusters
            pi = jnp.array([0.5, 0.5])

            # mu: random points on the dataset, they gonna be the center
            self.key, subkey = jax.random.split(self.key)
            index = jax.random.choice(subkey, jnp.arange(N), shape=(2,), replace=False)
            mu = X[index]

            # sigma: we initialize identity matrix for both sigmas
            sigma = jnp.stack([jnp.eye(d), jnp.eye(d)])

            log_likelihood_old = -jnp.inf

            for i in range(iterations):
                # step 2 Calculate resposibilities
                gamma, log_likelihood = self.expectation_step(X, pi, mu, sigma)

                # convergence condition: if the change is minimum, we're done
                diff = jnp.abs(log_likelihood - log_likelihood_old)
                if diff < tolerance:
                    print(f"Convergence reach at iteration {i+1} | Log-Likelihood: {log_likelihood:.4f}")
                    break

                log_likelihood_old = log_likelihood

                # step3: update parameters
                pi, mu, sigma = self.maximization_step(X, gamma)

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_pi = pi
                best_mu = mu
                best_sigma = sigma

        self.pi = best_pi
        self.mu = best_mu
        self.sigma = best_sigma

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
