import os
import pandas as pd
import jax
import jax.numpy as jnp
import mlflow
import matplotlib.pyplot as plt
import pickle
import linear
import logistic
from MLP_jax import Multilayer_Perceptron_JAX as MPJ
import classification_tree as tree
import em_algorithm as em
import naive_bayes as nb
import adaboost as ab
# EM algorithm and Naive-Bayes as mixture model


def load_data(csv, seed=73):
    """
    Extract and separate the data from the dataset

    I delete a lot of data, because there are flags used for astrophysics to confirm
    the exoplanets, confirmed false positive and false negatives. If I keep them, I
    would have a very high precission becasuse I would be using confirmed data (and
    it's not the purpose of the models)

    :param self: Description
    :return: transposed matrix (dxN)
    """
    exoplanets = pd.read_csv(csv)
    # 1 if is exoplanet, 0 otherwise
    exoplanets['target'] = exoplanets['koi_disposition'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)

    # Confirmed IDs, errors, data leakage
    col_delete = ['koi_disposition', 'koi_pdisposition', 'rowid', 'kepid',
                  'kepoi_name', 'kepler_name', 'koi_tce_delivname', 'koi_score']

    # detects the errors and flags
    flags = [col for col in exoplanets.columns if 'fpflag' in col]
    errors = [col for col in exoplanets.columns if 'err' in col]

    X_exoplanets = exoplanets.drop(columns=col_delete + flags + errors)
    X_exoplanets = X_exoplanets.dropna()

    y_og = X_exoplanets.pop('target').values
    X_og = X_exoplanets.values

    X = jnp.array(X_og)
    y = jnp.array(y_og)

    # Shuffle data
    seed = jax.random.PRNGKey(seed)
    index = jnp.arange(len(X))
    index_shuffle = jax.random.permutation(seed, index)

    X_shuffle = X[index_shuffle]
    y_shuffle = y[index_shuffle]

    # Separate data (Train: 70%, Validation: 15%, Test: 15%)
    n_total = len(X_shuffle)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    X_train = X_shuffle[:train_end]
    y_train = y_shuffle[:train_end]

    X_val = X_shuffle[train_end:val_end]
    y_val = y_shuffle[train_end:val_end]

    X_test = X_shuffle[val_end:]
    y_test = y_shuffle[val_end:]

    # normalize data
    # mean and deviation are calculated over training data only
    mean = jnp.mean(X_train, axis=0)
    std = jnp.std(X_train, axis=0)

    # prevent divission by zero
    std = jnp.where(std == 0, 1e-8, std)

    # Apply z = (x - u) / s
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    # logistic expects characteristic data as {X: d x N, W: k-1 x d}
    return (X_train_scaled.T, y_train, X_val_scaled.T, y_val, X_test_scaled.T, y_test, mean, std)


def plot_cm(ax, cm, fig, title):
    """
    Plot the confusion matrix

    Args:
        ax (_type_): _description_
        cm (_type_): _description_
        fig (_type_): _description_
        title (_type_): _description_
    """
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    tick_marks = jnp.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Real Label')

    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_mean, scaler_std = load_data('cumulative.csv')

    # configure mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("NASA_Exoplanet_Classification")

    # save normalization data
    jnp.save("scaler_mean.npy", jnp.array(scaler_mean))
    jnp.save("scaler_std.npy", jnp.array(scaler_std))

    # ________________  LINEAR ________________
    print("="*80)
    print(" "*30 + "LINEAR MODEL")
    print("="*80)

    with mlflow.start_run(run_name="Linear_Classifier_OLS"):
        print('Training lineal model')
        lin_model = linear.linear()
        lin_model.fit(X_train, y_train)
        print('training complete\n')

        # register hyperparameters
        mlflow.log_param("model_type", "Ordinary Least Squares")
        mlflow.log_param("threshold", 0.5)

        print('Generating predictions for validation')
        y_hat_val_lin = lin_model.estimate(X_val)
        continuous_val = lin_model.estimate_continuous(X_val)
        format_values = [round(float(v), 2) for v in continuous_val[:15]]

        # print(f"Decimal values: {format_values}")
        print(f"Model prediction:  {y_hat_val_lin[:15]}")
        print(f"Real values:       {y_val[:15]}")

        val_prec_lin, val_rec_lin, val_acc_lin, val_f1_lin, val_cm_lin = lin_model.calculate_metrics(y_val, y_hat_val_lin)  # noqa
        print("_"*80)
        print(f"Precision: {val_prec_lin:.4f} | Recall: {val_rec_lin:.4f} | Accuracy: {val_acc_lin:.4f} | F1-Score: {val_f1_lin:.4f}")  # noqa
        print("_"*80 + '\n')

        # Register validation data
        mlflow.log_metric("val_precision", val_prec_lin)
        mlflow.log_metric("val_recall", val_rec_lin)
        mlflow.log_metric("val_accuracy", val_acc_lin)
        mlflow.log_metric("val_f1", val_f1_lin)

        print('Generating predictions for test')
        y_hat_test_lin = lin_model.estimate(X_test)
        test_prec_lin, test_rec_lin, test_acc_lin, test_f1_lin, test_cm_lin = lin_model.calculate_metrics(y_test, y_hat_test_lin)  # noqa
        print("_"*80)
        print(f"Precision: {test_prec_lin:.4f} | Recall: {test_rec_lin:.4f} | Accuracy: {test_acc_lin:.4f} | F1-Score: {test_f1_lin:.4f}")  # noqa
        print("_"*80 + '\n')

        # Register test data
        mlflow.log_metric("test_precision", test_prec_lin)
        mlflow.log_metric("test_recall", test_rec_lin)
        mlflow.log_metric("test_accuracy", test_acc_lin)
        mlflow.log_metric("test_f1", test_f1_lin)

        # save beta
        jnp.save("linear_beta.npy", jnp.array(lin_model.beta))
        mlflow.log_artifact("linear_beta.npy")

        fig_val_lin, ax_val_lin = plt.subplots(figsize=(6, 5))
        plot_cm(ax_val_lin, val_cm_lin, fig_val_lin, "Confusion matrix Val - Lineal")
        fig_val_lin.savefig("cm_val_lineal.png", bbox_inches='tight')
        mlflow.log_artifact("cm_val_lineal.png")
        plt.close(fig_val_lin)

        fig_lin, ax_lin = plt.subplots(figsize=(6, 5))
        plot_cm(ax_lin, test_cm_lin, fig_lin, "Confusion matrix test - Lineal")
        fig_lin.savefig("cm_test_lineal.png", bbox_inches='tight')
        mlflow.log_artifact("cm_test_lineal.png")
        plt.close(fig_lin)

    # ________________  LOGISTIC  ________________
    print('\n')
    print("="*80)
    print(" "*30 + "LOGISTIC MODEL")
    print("="*80)
    print('Training logistic model')
    with mlflow.start_run(run_name="Logistic_Regression_Newton"):
        log_model = logistic.logistic()
        log_model.fit(X_train, y_train)
        print('training complete\n')

        # register hyperparameters
        mlflow.log_param("model_type", "Newton-Raphson")
        mlflow.log_param("max_iterations", 1000)
        mlflow.log_param("tolerance", 1e-3)

        print('Generating predictions for validation')
        y_hat_val_log = log_model.estimate(X_val)
        print(f"Model prediction: {y_hat_val_log[:15]}")
        print(f"Real values:      {y_val[:15]}")

        val_prec_log, val_rec_log, val_acc_log, val_f1_log, val_cm_log = log_model.calculate_metrics(y_val, y_hat_val_log)  # noqa
        print("_"*80)
        print(f"Precision: {val_prec_log:.4f} | Recall: {val_rec_log:.4f} | Accuracy: {val_acc_log:.4f} | F1-Score: {val_f1_log:.4f}")  # noqa
        print("_"*80 + '\n')

        # Register validation data
        mlflow.log_metric("val_precision", val_prec_log)
        mlflow.log_metric("val_recall", val_rec_log)
        mlflow.log_metric("val_accuracy", val_acc_log)
        mlflow.log_metric("val_f1", val_f1_log)

        print('Generating predictions for test')
        y_hat_test_log = log_model.estimate(X_test)
        test_prec_log, test_rec_log, test_acc_log, test_f1_log, test_cm_log = log_model.calculate_metrics(y_test, y_hat_test_log)  # noqa
        print("_"*80)
        print(f"Precision: {test_prec_log:.4f} | Recall: {test_rec_log:.4f} | Accuracy: {test_acc_log:.4f} | F1-Score: {test_f1_log:.4f}")  # noqa
        print("_"*80 + '\n')

        # Register test data
        mlflow.log_metric("test_precision", test_prec_log)
        mlflow.log_metric("test_recall", test_rec_log)
        mlflow.log_metric("test_accuracy", test_acc_log)
        mlflow.log_metric("test_f1", test_f1_log)

        # Save weights W
        jnp.save("logistic_W.npy", jnp.array(log_model.W))
        mlflow.log_artifact("logistic_W.npy")

        fig_val_log, ax_log = plt.subplots(figsize=(6, 5))
        plot_cm(ax_log, val_cm_log, fig_val_log, "Confusion matrix val - Logistic")
        fig_val_log.savefig("cm_val_log.png", bbox_inches='tight')
        mlflow.log_artifact("cm_val_log.png")
        plt.close(fig_val_log)

        fig_log, ax_log = plt.subplots(figsize=(6, 5))
        plot_cm(ax_log, test_cm_log, fig_log, "Confusion matrix test - Logistic")
        fig_log.savefig("cm_test_log.png", bbox_inches='tight')
        mlflow.log_artifact("cm_test_log.png")
        plt.close(fig_log)

    # ________________  MLP  ________________
    print('\n')
    print("="*80)
    print(" "*30 + "MLP Autodiff")
    print("="*80)
    print('MLP')
    with mlflow.start_run(run_name="MLP_autodiff"):
        jax_mlp = MPJ(X_train.T)
        jax_result, jax_loss_history = jax_mlp.fit_mlp_jax(X_train.T, y_train, 1000, 0.1)

        mlflow.log_param("model_type", "Autodiff")
        mlflow.log_param("epochs", 1000)
        mlflow.log_param("learning_rate", 0.1)

        # loss curve
        for step_num, loss_val in enumerate(jax_loss_history):
            mlflow.log_metric("loss", float(loss_val), step=step_num)

        print('Generating predictions for val')
        val_cm_jax, val_p_jax, val_r_jax, val_acc_mlp, val_f1_jax = jax_mlp.get_metrics(X_val.T, y_val)

        # Register validation data
        mlflow.log_metric("val_precision", val_p_jax)
        mlflow.log_metric("val_recall", val_r_jax)
        mlflow.log_metric("val_accuracy", val_acc_mlp)
        mlflow.log_metric("val_f1", val_f1_jax)

        print('Generating predictions for test')
        test_cm_jax, test_p_jax, test_r_jax, test_acc_mlp, test_f1_jax = jax_mlp.get_metrics(X_test.T, y_test)

        # Register validation data
        mlflow.log_metric("test_precision", test_p_jax)
        mlflow.log_metric("test_recall", test_r_jax)
        mlflow.log_metric("test_accuracy", test_acc_mlp)
        mlflow.log_metric("test_f1", test_f1_jax)

        fig_val_mlp, ax_mlp = plt.subplots(figsize=(6, 5))
        plot_cm(ax_mlp, val_cm_jax, fig_val_mlp, "Confusion matrix val - MLP")
        fig_val_mlp.savefig("cm_val_mlp.png", bbox_inches='tight')
        mlflow.log_artifact("cm_val_mlp.png")
        plt.close(fig_val_mlp)

        fig_mlp, ax_mlp = plt.subplots(figsize=(6, 5))
        plot_cm(ax_mlp, test_cm_jax, fig_mlp, "Confusion matrix test- MLP")
        fig_mlp.savefig("cm_test_mlp.png", bbox_inches='tight')
        mlflow.log_artifact("cm_test_mlp.png")
        plt.close(fig_mlp)

        # save trained data
        jnp.savez("mlp_params.npz", **jax_result)
        mlflow.log_artifact("mlp_params.npz")

    # ________________  Decision tree  ________________
    print('\n')
    print("="*80)
    print(" "*30 + "DECISION TREE MODEL")
    print("="*80)
    with mlflow.start_run(run_name="Decision_Tree_CART"):
        tree_model = tree.DecisionTree(max_depth=5, min_samples=10)

        print('Training Decision Tree model...')
        tree_model.fit(X_train.T, y_train)

        mlflow.log_param("model_type", "Decision Tree (CART)")
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("min_samples", 10)

        # Validation
        y_hat_val_tree = tree_model.predict(X_val.T)
        val_prec_tree, val_rec_tree, val_acc_tree, val_f1_tree, val_cm_tree = tree_model.calculate_metrics(y_val, y_hat_val_tree)  # noqa

        mlflow.log_metric("val_precision", val_prec_tree)
        mlflow.log_metric("val_recall", val_rec_tree)
        mlflow.log_metric("val_accuracy", val_acc_tree)
        mlflow.log_metric("val_f1", val_f1_tree)

        # Test
        print('Generating predictions for test')
        y_hat_test_tree = tree_model.predict(X_test.T)
        test_prec_tree, test_rec_tree, test_acc_tree, test_f1_tree, test_cm_tree = tree_model.calculate_metrics(y_test, y_hat_test_tree)  # noqa

        mlflow.log_metric("test_precision", test_prec_tree)
        mlflow.log_metric("test_recall", test_rec_tree)
        mlflow.log_metric("test_accuracy", test_acc_tree)
        mlflow.log_metric("test_f1", test_f1_tree)

        # validation
        fig_val_tree, ax_val_tree = plt.subplots(figsize=(6, 5))
        plot_cm(ax_val_tree, val_cm_tree, fig_val_tree, "Confusion matrix val - Tree")
        fig_val_tree.savefig("cm_val_tree.png", bbox_inches='tight')
        mlflow.log_artifact("cm_val_tree.png")
        plt.close(fig_val_tree)

        #  Test
        fig_test_tree, ax_test_tree = plt.subplots(figsize=(6, 5))
        plot_cm(ax_test_tree, test_cm_tree, fig_test_tree, "Confusion matrix test - Tree")
        fig_test_tree.savefig("cm_test_tree.png", bbox_inches='tight')
        mlflow.log_artifact("cm_test_tree.png")
        plt.close(fig_test_tree)

        # save tree
        with open("tree_model.pkl", "wb") as f:
            pickle.dump(tree_model, f)
        mlflow.log_artifact("tree_model.pkl")

    # ________________ Mixture Model  ________________
    print('\n')
    print("="*80)
    print(" "*30 + "MIXTURE MODEL (EM & NB)")
    print("="*80)
    with mlflow.start_run(run_name="Mixture_Model_EM_NB"):
        print('Training EM algorithm to find clusters...')
        # Initialice and train Expectation-Maximization
        em_model = em.em_algorithm(seed=73)
        # Use X_train.T for a vector form (N, d)
        em_model.fit_em(X_train.T, iterations=100, tolerance=1e-4, n_init=7)
        print('EM training complete\n')

        mlflow.log_param("model_type", "Mixture_Model_EM_NB")
        mlflow.log_param("em_iterations", 100)
        mlflow.log_param("em_tolerance", 1e-4)

        # VALIDATION
        print('Generating predictions for validation')
        nb_val = nb.naive_bayes(em_model.pi, em_model.mu, em_model.sigma, X_val.T)
        nb_val.model()  # generate self.y_pred
        y_hat_val_em = nb_val.y_pred

        # Extract metrics
        val_prec_em, val_rec_em, val_acc_em, val_f1_em = nb_val.calculate_metrics(y_val, y_hat_val_em)

        # Cconfusion matrix
        val_cm_em = jnp.array([
            [jnp.sum((y_hat_val_em == 0) & (y_val == 0)),
             jnp.sum((y_hat_val_em == 1) & (y_val == 0))],
            [jnp.sum((y_hat_val_em == 0) & (y_val == 1)),
             jnp.sum((y_hat_val_em == 1) & (y_val == 1))]])

        print("_"*80)
        print(f"Precision: {val_prec_em:.4f} | Recall: {val_rec_em:.4f} | Accuracy: {val_acc_em:.4f} | F1-Score: {val_f1_em:.4f}")  # noqa
        print("_"*80 + '\n')

        mlflow.log_metric("val_precision", val_prec_em)
        mlflow.log_metric("val_recall", val_rec_em)
        mlflow.log_metric("val_accuracy", val_acc_em)
        mlflow.log_metric("val_f1", val_f1_em)

        # TEST
        print('Generating predictions for test')
        nb_test = nb.naive_bayes(em_model.pi, em_model.mu, em_model.sigma, X_test.T)
        nb_test.model()
        y_hat_test_em = nb_test.y_pred

        test_prec_em, test_rec_em, test_acc_em, test_f1_em = nb_test.calculate_metrics(y_test, y_hat_test_em)

        test_cm_em = jnp.array([
            [jnp.sum((y_hat_test_em == 0) & (y_test == 0)),
             jnp.sum((y_hat_test_em == 1) & (y_test == 0))],
            [jnp.sum((y_hat_test_em == 0) & (y_test == 1)),
             jnp.sum((y_hat_test_em == 1) & (y_test == 1))]])

        print("_"*80)
        print(f"Precision: {test_prec_em:.4f} | Recall: {test_rec_em:.4f} | Accuracy: {test_acc_em:.4f} | F1-Score: {test_f1_em:.4f}")  # noqa
        print("_"*80 + '\n')

        mlflow.log_metric("test_precision", test_prec_em)
        mlflow.log_metric("test_recall", test_rec_em)
        mlflow.log_metric("test_accuracy", test_acc_em)
        mlflow.log_metric("test_f1", test_f1_em)

        # ARTIFACTS
        # save parameters learned by em algorithm
        jnp.savez("em_parameters.npz", pi=em_model.pi, mu=em_model.mu, sigma=em_model.sigma)
        mlflow.log_artifact("em_parameters.npz")

        fig_val_em, ax_val_em = plt.subplots(figsize=(6, 5))
        plot_cm(ax_val_em, val_cm_em, fig_val_em, "Confusion matrix val - EM+NB")
        fig_val_em.savefig("cm_val_em.png", bbox_inches='tight')
        mlflow.log_artifact("cm_val_em.png")
        plt.close(fig_val_em)

        fig_test_em, ax_test_em = plt.subplots(figsize=(6, 5))
        plot_cm(ax_test_em, test_cm_em, fig_test_em, "Confusion matrix test - EM+NB")
        fig_test_em.savefig("cm_test_em.png", bbox_inches='tight')
        mlflow.log_artifact("cm_test_em.png")
        plt.close(fig_test_em)

    # ________________ Mixture Model  ________________
    print('\n')
    print("="*80)
    print(" "*30 + "ADABOOST MODEL")
    print("="*80)
    with mlflow.start_run(run_name="AdaBoost_Stumps"):
        boost_model = ab.AdaBoost(M=50, seed=73)

        print('Training AdaBoost model...')
        boost_model.fit(X_train.T, y_train)

        mlflow.log_param("model_type", "AdaBoost (M1)")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("base_estimator", "Decision Stump")

        # Validation
        print('Generating predictions for validation')
        y_hat_val_boost = boost_model.predict(X_val.T)
        val_prec_bst, val_rec_bst, val_acc_bst, val_f1_bst = boost_model.calculate_metrics(y_val, y_hat_val_boost)[:4]
        
        # Confusion matrix
        val_cm_ab = jnp.array([
            [jnp.sum((y_hat_val_boost == 0) & (y_val == 0)), jnp.sum((y_hat_val_boost == 1) & (y_val == 0))],
            [jnp.sum((y_hat_val_boost == 0) & (y_val == 1)), jnp.sum((y_hat_val_boost == 1) & (y_val == 1))]
        ])

        print("_"*80)
        print(f"Precision: {val_prec_bst:.4f} | Recall: {val_rec_bst:.4f} | Accuracy: {val_acc_bst:.4f} | F1-Score: {val_f1_bst:.4f}")
        print("_"*80 + '\n')

        mlflow.log_metric("val_precision", val_prec_bst)
        mlflow.log_metric("val_recall", val_rec_bst)
        mlflow.log_metric("val_accuracy", val_acc_bst)
        mlflow.log_metric("val_f1", val_f1_bst)

        # Test
        print('Generating predictions for test')

        y_hat_test_boost = boost_model.predict(X_val.T)
        test_prec_bst, test_rec_bst, test_acc_bst, test_f1_bst = boost_model.calculate_metrics(y_test, y_hat_test_boost)[:4]
        
        # Confusion matrix
        test_cm_ab = jnp.array([
            [jnp.sum((y_hat_test_boost == 0) & (y_test == 0)), jnp.sum((y_hat_test_boost == 1) & (y_test == 0))],
            [jnp.sum((y_hat_test_boost == 0) & (y_test == 1)), jnp.sum((y_hat_test_boost == 1) & (y_test == 1))]
        ])

        print("_"*80)
        print(f"Precision: {test_prec_bst:.4f} | Recall: {test_rec_bst:.4f} | Accuracy: {test_acc_bst:.4f} | F1-Score: {test_f1_bst:.4f}")
        print("_"*80 + '\n')

        mlflow.log_metric("val_precision", test_prec_bst)
        mlflow.log_metric("val_recall", test_rec_bst)
        mlflow.log_metric("val_accuracy", test_acc_bst)
        mlflow.log_metric("val_f1", test_f1_bst)

        # ARTIFACTS
        # save parameters learned by em algorithm
        jnp.savez("ab_parameters.npz", pi=boost_model.pi, mu=boost_model.mu, sigma=boost_model.sigma)
        mlflow.log_artifact("ab_parameters.npz")

        fig_val_ab, ax_val_ab = plt.subplots(figsize=(6, 5))
        plot_cm(ax_val_ab, val_cm_ab, fig_val_ab, "Confusion matrix val - AdaBoost")
        fig_val_ab.savefig("cm_val_ab.png", bbox_inches='tight')
        mlflow.log_artifact("cm_val_ab.png")
        plt.close(fig_val_ab)

        fig_test_ab, ax_test_ab = plt.subplots(figsize=(6, 5))
        plot_cm(ax_test_ab, test_cm_ab, fig_test_ab, "Confusion matrix test - AdaBoost")
        fig_test_ab.savefig("cm_test_ab.png", bbox_inches='tight')
        mlflow.log_artifact("cm_test_ab.png")
        plt.close(fig_test_ab)


if __name__ == "__main__":
    main()
