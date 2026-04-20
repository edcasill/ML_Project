import jax.numpy as jnp


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    """
    Use shannon impurity as splitting criterion
    """
    def __init__(self, max_depth=5, min_samples=10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.num_classes = 2

    def entropy(self, y):
        counts = jnp.bincount(y, length=self.num_classes)
        probs = counts / len(y)
        probs = probs[probs > 0]  # Avoid log(0)
        return -jnp.sum(probs * jnp.log2(probs))

    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        n_samples, n_features = X.shape
        current_entropy = self.entropy(y)

        for feature in range(n_features):
            threshold = jnp.mean(X[:, feature])
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if not jnp.any(left_mask) or not jnp.any(right_mask):
                continue

            y_l, y_r = y[left_mask], y[right_mask]
            n_l, n_r = len(y_l), len(y_r)

            entropy_l, entropy_r = self.entropy(y_l), self.entropy(y_r)
            gain = current_entropy - (n_l / n_samples) * entropy_l - (n_r / n_samples) * entropy_r
            if gain > best_gain:
                best_gain = gain
                split_idx = feature
                split_thresh = threshold
        return split_idx, split_thresh

    def build_tree(self, X, y, depth=0):
        n_samples = len(y)
        counts = jnp.bincount(y, length=self.num_classes)
        most_common = jnp.argmax(counts)

        if depth >= self.max_depth or n_samples < self.min_samples or jnp.max(counts) == n_samples:
            return Node(value=most_common)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return Node(value=most_common)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        return self.predict_sample(x, node.right)

    def predict(self, X):
        return jnp.array([self.predict_sample(x, self.root) for x in X])

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate metrix and return confusion matrix
        """
        cm_1d = jnp.bincount(y_true * self.num_classes + y_pred, length=self.num_classes**2)
        cm = cm_1d.reshape((self.num_classes, self.num_classes))

        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]

        epsilon = 1e-7
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        return float(precision), float(recall), float(accuracy), float(f1), cm
