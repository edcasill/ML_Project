import os
import pandas as pd
import jax
import jax.numpy as jnp
import linear
import logistic


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

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")


if __name__ == "__main__":
    main()
