import jax
import jax.numpy as jnp
import classification_tree as tree


class AdaBoost:
    def __init__(self, M=50, seed=73):
        self.key = jax.random.PRNGKey(seed)
        self.M = M  # quantity of estimators or boosting rounds
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        """


        Args:
            X: (N, d) characteristic matrix
            y: (N, ) labels {0, 1}
        """
        N = X.shape[0]

        # step 1: initialize weights -> {w_i} to 1/N
        w = jnp.ones(N) / N

        # map my labels to {-1, 1} labels
        y_boost = jnp.where(y == 0, -1, 1)  # if y=0 -> -1, 1 otherwise

        for m in range(self.M):
            self.key, subkey = jax.random.split(self.key)  # for a random choice

            # re-sample based on weigths w
            index = jax.random.choice(subkey, jnp.arange(N), shape=(N,), p=w, replace=True)
            X_m = X[index]
            y_m = y[index]

            # initialize weak learner (stump)
            stump = tree.DecisionTree(max_depth=1, min_samples=2)
            stump.fit(X_m, y_m)

            # predict original data
            predict = stump.predict(X)
            # convert {0, 1} tree labels into {-1, 1} labels
            predict_boost = jnp.where(predict == 0, -1, 1)

            # step 2: calculate error
            indicator = jnp.where(predict_boost != y_boost, 1, 0)
            err_m = jnp.sum(w * indicator) / jnp.sum(w)
            err_m = jnp.clip(err_m, 1e-10, 1 - 1e-10)  # avoid 0 division

            # step 3: calculate alpha
            alpha_m = 0.5 * jnp.log((1 - err_m) / err_m)

            # update the weights
            w = w * jnp.exp(-alpha_m * y_boost * predict_boost)
            w = w / jnp.sum(w)  # normalize weigths

            # save model and weigths
            self.models.append(stump)
            self.alphas.append(alpha_m)

            # if there's no error on the model we stop it
            if err_m == 0:
                break
    
    def prediction(self, X):
        """
        """
        # save predictions from all stumps
        stump_predict = jnp.zeros((X.shape[0], len(self.models)))

        for index, (alpha, stump) in enumerate(zip(self.alphas, self.models)):
            pred = stump.prediction(X)
            pred_boost = jnp.where(pred == 0, -1, 1)
            # acumulate the prediction (alpha * prediction)
            stump_predict = stump_predict.at[:, index].set(alpha * pred_boost)

        final_sum = jnp.sum(stump_predict, axis=1)  # ponderate sumatory
        final_predict = jnp.sign(final_sum)

        # return labels to {0, 1}
        og_labels = jnp.where(final_predict == -1, 0, 1)

        return og_labels
    
    def calculate_metrics(self, y, y_hat):   
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
