import os
import pandas as pd
import jax
import jax.numpy as jnp
import linear
import logistic
import mlflow


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


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_mean, scaler_std = load_data('cumulative.csv')

    # configure mlflow
    mlflow.set_experiment("NASA_Exoplanet_Classification")

    # save normalization data
    jnp.save("scaler_mean.npy", jnp.array(scaler_mean))
    jnp.save("scaler_std.npy", jnp.array(scaler_std))

    print("="*80)
    print(" "*30 + "LINEAR MODEL")
    print("="*80)

    with mlflow.start_run(run_name="Linear_Classifier_OLS"):
        print('Training lineal model')
        lin_model = linear.linear()
        lin_model.fit(X_train, y_train)
        print('training complete')

        # register hyperparameters
        mlflow.log_param("model_type", "Ordinary Least Squares")
        mlflow.log_param("threshold", 0.5)

        print('Generating predictions for validation')
        y_hat_val_lin = lin_model.estimate(X_val)
        continuous_val = lin_model.estimate_continuous(X_val)
        format_values = [round(float(v), 2) for v in continuous_val[:15]]

        # print(f"Decimal values: {format_values}")
        print(f"Model prediction:  {y_hat_val_lin[:15]}") 
        print(f"Real values:       {y_val[:15]}\n")
        
        val_prec_lin, val_rec_lin, val_f1_lin = lin_model.calculate_metrics(y_val, y_hat_val_lin)
        print("_"*60)
        print(f"Precision: {val_prec_lin:.4f} | Recall: {val_rec_lin:.4f} | F1-Score: {val_f1_lin:.4f}")

        # Register validation data
        mlflow.log_metric("val_precision", val_prec_lin)
        mlflow.log_metric("val_recall", val_rec_lin)
        mlflow.log_metric("val_f1", val_f1_lin)
        
        print('Generating predictions for test')
        y_hat_test_lin = lin_model.estimate(X_test)
        test_prec_lin, test_rec_lin, test_f1_lin = lin_model.calculate_metrics(y_test, y_hat_test_lin)
        print("_"*60)
        print(f"Precision: {test_prec_lin:.4f} | Recall: {test_rec_lin:.4f} | F1-Score: {test_f1_lin:.4f}\n")

        # Register test data
        mlflow.log_metric("test_precision", test_prec_lin)
        mlflow.log_metric("test_recall", test_rec_lin)
        mlflow.log_metric("test_f1", test_f1_lin)

        # save beta
        jnp.save("linear_beta.npy", jnp.array(lin_model.beta))
        mlflow.log_artifact("linear_beta.npy")

    ###################################################################################################
    print("="*80)
    print(" "*30 + "LOGISTIC MODEL")
    print("="*80)
    print('Training logistic model')
    with mlflow.start_run(run_name="Logistic_Regression_Newton"):
        log_model = logistic.logistic()
        log_model.fit(X_train, y_train)
        print('training complete')

        # register hyperparameters
        mlflow.log_param("model_type", "Newton-Raphson")
        mlflow.log_param("max_iterations", 1000)
        mlflow.log_param("tolerance", 1e-3)

        print('Generating predictions for validation')
        y_hat_val_log = log_model.estimate(X_val)
        print(f"Predicción del Modelo: {y_hat_val_log[:15]}")
        print(f"Realidad (Etiquetas):  {y_val[:15]}")

        val_prec_log, val_rec_log, val_f1_log = log_model.calculate_metrics(y_val, y_hat_val_log)
        print("_"*60)
        print(f"Precision: {val_prec_log:.4f} | Recall: {val_rec_log:.4f} | F1-Score: {val_f1_log:.4f}")

        # Register validation data
        mlflow.log_metric("val_precision", val_prec_log)
        mlflow.log_metric("val_recall", val_rec_log)
        mlflow.log_metric("val_f1", val_f1_log)

        print('Generating predictions for test')
        y_hat_test_log = log_model.estimate(X_test)
        test_prec_log, test_rec_log, test_f1_log = log_model.calculate_metrics(y_test, y_hat_test_log)
        print("_"*60)
        print(f"Precision: {test_prec_log:.4f} | Recall: {test_rec_log:.4f} | F1-Score: {test_f1_log:.4f}\n")

        # Register test data
        mlflow.log_metric("test_precision", test_prec_log)
        mlflow.log_metric("test_recall", test_rec_log)
        mlflow.log_metric("test_f1", test_f1_log)

        # Save weights W
        jnp.save("logistic_W.npy", jnp.array(log_model.W))
        mlflow.log_artifact("logistic_W.npy")


if __name__ == "__main__":
    main()
