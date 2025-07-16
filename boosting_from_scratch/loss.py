# boosting_from_scratch/loss.py

import numpy as np

# Helper for LogLoss (sigmoid function)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- MSE Loss for Regression ---
class MSELoss:
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def gradient(self, y_true, y_pred):
        # Gradient of MSE: 2 * (y_pred - y_true) / N
        # In GBM, we fit to negative gradient, so (y_true - y_pred)
        # We'll use (y_pred - y_true) here, and take negative in booster.
        return y_pred - y_true

    def hessian(self, y_true, y_pred):
        # Hessian of MSE is constant: 1
        return np.ones_like(y_true)

# --- LogLoss for Binary Classification ---
class LogLoss:
    def loss(self, y_true, y_pred_raw):
        # y_pred_raw are the raw outputs of the ensemble (before sigmoid)
        # Convert y_true to {0, 1} if it's not already
        y_true_binary = (y_true > 0.5).astype(int) # Ensure binary 0/1

        # Clamp predictions to avoid log(0) or log(1) issues
        p = sigmoid(y_pred_raw)
        p = np.clip(p, 1e-10, 1 - 1e-10) # Clip probabilities to avoid log(0)

        return -np.mean(y_true_binary * np.log(p) + (1 - y_true_binary) * np.log(1 - p))

    def gradient(self, y_true, y_pred_raw):
        # Gradient of LogLoss: sigmoid(y_pred_raw) - y_true
        # y_true should be binary (0 or 1)
        y_true_binary = (y_true > 0.5).astype(int) # Ensure binary 0/1
        return sigmoid(y_pred_raw) - y_true_binary

    def hessian(self, y_true, y_pred_raw):
        # Hessian of LogLoss: sigmoid(y_pred_raw) * (1 - sigmoid(y_pred_raw))
        p = sigmoid(y_pred_raw)
        return p * (1 - p)