"""
Gaussian Process Dynamics Predictor for Vehicle Dynamics
Provides uncertainty-aware predictions for LMPC
"""

import numpy as np
import pickle
import os
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings about kernel bounds – they are not fatal.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class GPDynamicsPredictor:
    """
    Gaussian Process-based dynamics predictor.
    Trains one GP per state dimension on (x_k, u_k) -> x_{k+1}.

    To keep training fast:
      - We subsample to at most max_train_points points.
      - We use small n_restarts_optimizer.
    """

    def __init__(
        self,
        state_dim: int = 6,
        input_dim: int = 2,
        noise_level: float = 0.05,
        max_train_points: int = 2000,
        n_restarts: int = 1,
        random_seed: int = 0,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.max_train_points = max_train_points
        self.n_restarts = n_restarts
        self.rng = np.random.RandomState(random_seed)

        # Storage for training data
        self.all_states = []
        self.all_actions = []
        self.all_next_states = []

        # Normalization parameters
        self.X_mean = None
        self.X_std = None

        # One GP per state dimension
        self.gp_models = []

        for _ in range(state_dim):
            # Use relatively broad bounds so optimizer isn't stuck at edges
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                + WhiteKernel(
                    noise_level=noise_level,
                    noise_level_bounds=(1e-8, 1e1),
                )
            )

            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                alpha=1e-6,
                normalize_y=True,
            )
            self.gp_models.append(gp)

        print(f"✓ Initialized {state_dim} Gaussian Process models")

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def add_trajectory(self, states: np.ndarray, actions: np.ndarray):
        """
        Add a trajectory (states, actions) to the training buffer.
        states:  (T, state_dim)
        actions: (T, input_dim)
        """
        T = len(states)
        for t in range(T - 1):
            self.all_states.append(states[t])
            self.all_actions.append(actions[t])
            self.all_next_states.append(states[t + 1])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, verbose: bool = True):
        """
        Train all Gaussian Process models on collected data.

        To keep runtime manageable, we subsample the dataset if it
        exceeds max_train_points.
        """
        if len(self.all_states) == 0:
            print("⚠ No training data available for GP!")
            return

        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        next_states = np.array(self.all_next_states)

        N = states.shape[0]
        if N > self.max_train_points:
            idx = self.rng.choice(N, self.max_train_points, replace=False)
            states = states[idx]
            actions = actions[idx]
            next_states = next_states[idx]
            if verbose:
                print(f"Subsampled GP training data: {N} → {self.max_train_points} points")

        # Input: [state, action]
        X = np.hstack([states, actions])

        # Normalize inputs
        if self.X_mean is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8

        X_norm = (X - self.X_mean) / self.X_std

        if verbose:
            print(f"\nTraining Gaussian Processes on {X_norm.shape[0]} samples...")
            print(f"Input dimension: {X_norm.shape[1]} "
                  f"(state={self.state_dim}, action={self.input_dim})")

        state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']

        for i, gp in enumerate(self.gp_models):
            y = next_states[:, i]
            if verbose:
                name = state_names[i] if i < len(state_names) else f"x[{i}]"
                print(f"  Training GP for {name}...", end=" ")
            try:
                gp.fit(X_norm, y)
                if verbose:
                    r2 = gp.score(X_norm, y)
                    print(f"R² = {r2:.4f} ✓")
            except Exception as e:
                if verbose:
                    print(f"Warning during GP fit: {str(e)[:70]}... (continuing)")

        if verbose:
            print("✓ All GPs trained!\n")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _normalize_input(self, state, action):
        X = np.hstack([state, action]).reshape(1, -1)
        if self.X_mean is not None:
            X = (X - self.X_mean) / self.X_std
        return X

    def predict(self, state, action, return_std: bool = True):
        """
        Predict next state given current state and action.
        Returns:
            mean_pred: (state_dim,)
            std_pred:  (state_dim,) if return_std=True
        """
        X = self._normalize_input(state, action)

        means, stds = [], []
        for gp in self.gp_models:
            try:
                if return_std:
                    m, s = gp.predict(X, return_std=True)
                    means.append(float(m[0]))
                    stds.append(float(s[0]))
                else:
                    m = gp.predict(X)
                    means.append(float(m[0]))
            except Exception:
                means.append(0.0)
                if return_std:
                    stds.append(1.0)

        mean_pred = np.array(means)
        if return_std:
            return mean_pred, np.array(stds)
        else:
            return mean_pred

    def predict_trajectory(self, initial_state, actions, return_std: bool = True):
        """Roll out the GP forward over a sequence of actions."""
        states = [np.array(initial_state)]
        stds_list = [np.zeros(self.state_dim)]

        current = np.array(initial_state)
        for u in actions:
            if return_std:
                nxt, std = self.predict(current, u, return_std=True)
                stds_list.append(std)
            else:
                nxt = self.predict(current, u, return_std=False)
            states.append(nxt)
            current = nxt

        if return_std:
            return np.array(states), np.array(stds_list)
        else:
            return np.array(states)

    def get_uncertainty_map(self, test_states, test_actions):
        """Average GP std over state dimensions for each (x,u) pair."""
        uncertainties = []
        for x, u in zip(test_states, test_actions):
            _, std = self.predict(x, u, return_std=True)
            uncertainties.append(float(np.mean(std)))
        return np.array(uncertainties)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_model(self, filepath: str):
        data = {
            "gp_models": self.gp_models,
            "state_dim": self.state_dim,
            "input_dim": self.input_dim,
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "all_states": self.all_states,
            "all_actions": self.all_actions,
            "all_next_states": self.all_next_states,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"✓ GP models saved to {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"⚠ Model file not found: {filepath}")
            return
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.gp_models = data["gp_models"]
        self.state_dim = data["state_dim"]
        self.input_dim = data["input_dim"]
        self.X_mean = data.get("X_mean", None)
        self.X_std = data.get("X_std", None)
        self.all_states = data.get("all_states", [])
        self.all_actions = data.get("all_actions", [])
        self.all_next_states = data.get("all_next_states", [])
        print(f"✓ GP models loaded from {filepath}")


if __name__ == "__main__":
    # Small self-test (optional)
    print("=" * 60)
    print("Testing Gaussian Process Dynamics Predictor")
    print("=" * 60 + "\n")

    gp = GPDynamicsPredictor(state_dim=6, input_dim=2, noise_level=0.01, max_train_points=500)

    print("Generating dummy training data...")
    for _ in range(5):
        T = 50
        X = np.random.randn(T, 6) * 0.5
        U = np.random.randn(T, 2) * 0.3
        gp.add_trajectory(X, U)

    print("\nTraining Gaussian Processes...")
    gp.train(verbose=True)

    x_test = np.array([0.5, 0.1, 0.0, 0.0, 1.0, 0.0])
    u_test = np.array([0.1, 0.5])

    mean, std = gp.predict(x_test, u_test, return_std=True)
    print(f"\nInput state:  {x_test}")
    print(f"Input action: {u_test}")
    print(f"Predicted next state: {mean}")
    print(f"Uncertainty (std):    {std}\n")

    gp.save_model("test_gp_model.pkl")
    print("Done.")
