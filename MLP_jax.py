import jax
import jax.numpy as jnp


class Multilayer_Perceptron_JAX():
    """
    Multilayer pereptron ussing the JAX forward implementation with autodiff
    """
    def __init__(self, X, layers=[0, 64, 32, 1], seed=73):
        key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(key, len(layers) - 1)
        layers[0] = X.shape[1]  # input layer
        self.layers = layers
        self.params = {}

        # this is implemented if in a future we want to add more hiden layers
        for i in range(len(self.layers) - 1):
            dim_in = self.layers[i]
            dim_out = self.layers[i+1]
            
            # Weights and bias initialization for each hidden layer and the output layer
            self.params[f'W{i+1}'] = jax.random.normal(self.keys[i], (dim_in, dim_out)) * jnp.sqrt(2.0 / dim_in)
            self.params[f'b{i+1}'] = jnp.zeros((1, dim_out))

    @staticmethod
    def forward_propagation(params, X):
        """
        Get the first results from the mlp, passing the input values.

        Args:
            params (_type_): _description_
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        activations_hidden = X
        total_layers = len(params) // 2

        for i in range(1, total_layers): 
            Z_hidden = (activations_hidden @ params[f'W{i}']) + params[f'b{i}']
            activations_hidden = jnp.maximum(0, Z_hidden)

        # forward result on output layer
        Z_out = (activations_hidden @ params[f'W{total_layers}']) + params[f'b{total_layers}']
        y_out = 1 / (1 + jnp.exp(-Z_out)) # sigmoid
        return y_out.squeeze()
    
    @staticmethod
    def loss(params, X, Y):
        """
        Cost function (Cross-Entropy)
        """
        predictions = Multilayer_Perceptron_JAX.forward_propagation(params, X)
        epsilon = 1e-8
        predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)

        # Binary cross entropy
        error = -jnp.mean(Y * jnp.log(predictions) + (1 - Y) * jnp.log(1 - predictions))
        return error
    
    def get_metrics(self, X_test, Y_test):
        """
        Calculate metrics by evaluation of the model
        """
        pred_probs = self.forward_propagation(self.params, X_test)
        y_pred = jnp.where(pred_probs > 0.5, 1, 0)
        num_classes = 2

        cm_1d = jnp.bincount(Y_test * num_classes + y_pred, length=num_classes**2)
        cm = cm_1d.reshape((num_classes, num_classes))
        
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
        epsilon = 1e-7
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        accuracy = (TP + TN) / jnp.maximum(TP + TN + FP + FN, 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        return cm, precision, recall, accuracy, f1
    
    def fit_mlp_jax(self, X, Y, epochs, learning_rate):
        """
        Learning process of the MLP
        """
        # gradient function
        grad_fn = jax.grad(self.loss)

        loss_history = []

        for epoch in range(epochs):
            # get all grafients evaluating grad_fn
            gradients = grad_fn(self.params, X, Y)

            # update parameters
            total_layers = len(self.params) // 2
            for i in range(1, total_layers + 1):
                self.params[f'W{i}'] -= learning_rate * gradients[f'W{i}']
                self.params[f'b{i}'] -= learning_rate * gradients[f'b{i}']

            current_loss = self.loss(self.params, X, Y)
            loss_history.append(current_loss)

            if epoch % 10 == 0:
                y_out = self.forward_propagation(self.params, X)
                predictions = jnp.where(y_out > 0.5, 1, 0)
                precision = jnp.mean(predictions == Y)
                print(f"Epoch {epoch} | Precissionn: {precision:.4f} | Loss: {current_loss:.4f}")

        return self.params, loss_history