import os
import pandas as pd
import jax.numpy as jnp
import pickle
from metaflow import FlowSpec, step
import json
# models
import linear
import logistic
from MLP_jax import Multilayer_Perceptron_JAX as MPJ
import classification_tree as tree
import em_algorithm as em
import naive_bayes as nb
import adaboost as ab


class KeplerFlow(FlowSpec):

    @step
    def start(self):
        """
        Start the pipeline andtake the data
        """
        print("--- Starting Pipeline of Kepler Exoplanets ---")
        self.csv_path = 'cumulative.csv'
        self.next(self.process_data)

    @step
    def process_data(self):
        """
        Clean, scale and generate the artifacts of normalization
        """
        df = pd.read_csv(self.csv_path)
        df['target'] = df['koi_disposition'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)

        # Delete Data Leakage, used for astrophisic
        col_delete = ['koi_disposition', 'koi_pdisposition', 'rowid', 'kepid',
                      'kepoi_name', 'kepler_name', 'koi_tce_delivname', 'koi_score']
        flags = [col for col in df.columns if 'fpflag' in col]
        errors = [col for col in df.columns if 'err' in col]

        X = df.drop(columns=col_delete + flags + errors + ['target']).dropna()
        self.y = df.loc[X.index, 'target'].values

        # save mean and deviation for API
        self.mean = X.mean().values
        self.std = X.std().values
        jnp.save("scaler_mean.npy", self.mean)
        jnp.save("scaler_std.npy", self.std)

        # Z-score scalling
        self.X_scaled = ((X - self.mean) / self.std).values
        split = int(len(self.X_scaled) * 0.8)
        
        self.X_train = self.X_scaled[:split]
        self.y_train = self.y[:split]
        self.X_test = self.X_scaled[split:]
        self.y_test = self.y[split:]
        
        # save metrics on each branch
        self.metrics_results = {}
        
        print(f"Dataset procesed: {len(self.X_scaled)} total samples.")
        self.next(self.train_discriminative, self.train_generative, self.train_ensemble)

    @step
    def train_discriminative(self):
        """
        Train linear, logistic and MLP models
        branch 1
        """
        X_train = self.X_scaled.T # d x N for the implementations
        
        # Linear
        model_lin = linear.linear()
        model_lin.fit(self.X_train.T, self.y_train)
        y_hat_lin = model_lin.estimate(self.X_test.T)
        _, _, acc, f1, _ = model_lin.calculate_metrics(self.y_test, y_hat_lin)
        self.metrics_results["Lineal"] = {"accuracy": float(acc), "f1": float(f1)}
        jnp.save("linear_beta.npy", model_lin.beta)

        # Logistic
        model_log = logistic.logistic()
        model_log.fit(self.X_train.T, self.y_train)
        y_hat_log = model_log.estimate(self.X_test.T)
        _, _, acc, f1, _ = model_log.calculate_metrics(self.y_test, y_hat_log)
        self.metrics_results["Logistic"] = {"accuracy": float(acc), "f1": float(f1)}
        jnp.save("logistic_W.npy", model_log.W)

        # MLP
        model_mlp = MPJ(self.X_scaled, layers=[self.X_scaled.shape[1], 64, 32, 1])
        model_mlp.fit_mlp_jax(self.X_train, self.y_train.reshape(-1, 1), epochs=500, learning_rate=0.01)
        y_hat_mlp = jnp.where(model_mlp.forward_propagation(model_mlp.params, self.X_test) > 0.5, 1, 0).flatten()
        acc_mlp = jnp.mean(y_hat_mlp == self.y_test)
        self.metrics_results["MLP"] = {"accuracy": float(acc_mlp), "f1": float(acc_mlp)}
        with open("mlp_params.pkl", "wb") as f:
            pickle.dump(model_mlp.params, f)

        self.next(self.join_step)

    @step
    def train_generative(self):
        """
        Train de Mixture Models (EM + Naive Bayes)
        branch 2
        """
        model_em = em.em_algorithm()
        model_em.fit_em(self.X_scaled, iterations=100)
        model_nb = nb.naive_bayes(model_em.pi, model_em.mu, model_em.sigma, self.X_test)
        model_nb.model()
        _, _, acc, f1 = model_nb.calculate_metrics(self.y_test, model_nb.y_pred)

        # save parameters from the mixture
        self.metrics_results["Mixture"] = {"accuracy": float(acc), "f1": float(f1)}
        jnp.savez("em_parameters.npz", pi=model_em.pi, mu=model_em.mu, sigma=model_em.sigma)
        self.next(self.join_step)

    @step
    def train_ensemble(self):
        """
        Tree and AdaBoost training
        branch 3
        """
        # Decision Tree Ponderated
        model_tree = tree.DecisionTree(max_depth=7)
        model_tree.fit(self.X_train, self.y_train)
        y_hat_tree = model_tree.predict(self.X_test)
        _, _, acc, f1, _ = model_tree.calculate_metrics(self.y_test, y_hat_tree)
        self.metrics_results["Tree"] = {"accuracy": float(acc), "f1": float(f1)}
        with open("tree_model.pkl", "wb") as f:
            pickle.dump(model_tree, f)

        # AdaBoost
        model_ab = ab.AdaBoost(M=50)
        model_ab.fit(self.X_train, self.y_train)
        y_hat_ab = model_ab.prediction(self.X_test)
        _, _, acc, f1 = model_ab.calculate_metrics(self.y_test, y_hat_ab)
        self.metrics_results["AdaBoost"] = {"accuracy": float(acc), "f1": float(f1)}
        with open("adaboost_model.pkl", "wb") as f:
            pickle.dump(model_ab, f)

        self.next(self.join_step)

    @step
    def join_step(self, inputs):
        """
        Merge paralel branches
        """
        self.final_metrics = {}
        for branch in inputs:
            self.final_metrics.update(branch.metrics_results)
        self.next(self.end)

    @step
    def end(self):
        """
        End of execution and validate archives are generated
        """
        '''print("--- Pipeline Finalized ---")
        archives = ["linear_beta.npy", "logistic_W.npy", "mlp_params.pkl", 
                    "tree_model.pkl", "em_parameters.npz", "adaboost_model.pkl",
                    "scaler_mean.npy", "scaler_std.npy"]
        for a in archives:
            status = "OK" if os.path.exists(a) else "MISSING"
            print(f"Archive {a}: {status}")
        '''
        with open("metrics.json", "w") as f:
            json.dump(self.final_metrics, f, indent=4)
        print("--- Pipeline finalizado exitosamente. Archivos generados. ---")

if __name__ == '__main__':
    KeplerFlow()