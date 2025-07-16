# boosting_from_scratch/early_stopping.py

import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience    # Number of epochs to wait after last improvement
        self.min_delta = min_delta  # Minimum change to qualify as an improvement
        self.best_loss = np.inf     # Stores the best validation loss seen so far
        self.patience_counter = 0   # Counter for how many epochs without improvement
        self.stop_training = False  # Flag to signal when to stop

    def __call__(self, current_val_loss):
        # This method is called after each boosting round (or epoch)

        if current_val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = current_val_loss
            self.patience_counter = 0 # Reset counter
            self.stop_training = False
        else:
            # No significant improvement
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.stop_training = True # Patience limit reached

        return self.stop_training