# boosting_from_scratch/booster.py

import numpy as np
from sklearn.model_selection import train_test_split # Import for data splitting
from .tree import DecisionTree # Import your custom DecisionTree
from .loss import MSELoss, LogLoss, sigmoid # Import your custom Loss functions
from .early_stopping import EarlyStopping # Import EarlyStopping

class GradientBooster:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, loss_type='regressor', feature_types=None,
                 early_stopping_rounds=None, min_delta=0.0):
        
        if loss_type not in ['regressor', 'classifier']:
            raise ValueError("loss_type must be 'regressor' or 'classifier'")
            
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss_type = loss_type
        self.feature_types = feature_types # Pass feature types to individual trees
        
        self.early_stopping_rounds = early_stopping_rounds
        self.min_delta = min_delta

        self.estimators = [] # List to store the individual decision trees
        self.initial_prediction = None # Initial raw prediction (F0)
        
        # Select the appropriate loss function based on loss_type
        if self.loss_type == 'regressor':
            self.loss_fn = MSELoss()
        elif self.loss_type == 'classifier':
            self.loss_fn = LogLoss()
        
        # Placeholder for storing loss history (for plotting)
        self.train_loss_history = []
        self.val_loss_history = [] # To store validation loss history
        
        # Feature importances will be calculated and stored here
        self.feature_importances_ = None # Initialized as None, set to array in fit

    def fit(self, X, y, eval_set=None, validation_split=0.2):
        n_samples, n_features = X.shape

        self.estimators = [] # Clear estimators from previous fits
        self.train_loss_history = [] # Clear history
        self.val_loss_history = [] # Clear history
        self.feature_importances_ = np.zeros(n_features) # NEW: Initialize overall feature importances

        # Handle validation split or custom eval_set
        if eval_set is not None:
            X_train, y_train = X, y # Use full X,y as training
            X_val, y_val = eval_set[0], eval_set[1]
        elif validation_split > 0 and X.shape[0] > 1: # Ensure enough samples for split
            # Perform train-validation split if no eval_set provided
            # Stratify for classification to maintain class balance
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42,
                stratify=y if self.loss_type == 'classifier' and len(np.unique(y)) > 1 else None 
            )
        else: # No validation split or not enough samples
            X_train, y_train = X, y
            X_val, y_val = None, None # No validation set

        # Initialize raw predictions for training set (F0)
        if self.loss_type == 'regressor':
            self.initial_prediction = np.mean(y_train) # Use train_y for initial pred
        elif self.loss_type == 'classifier':
            # Ensure y_train is binary for initial prediction
            y_train_binary = (y_train > 0.5).astype(int)
            p_initial = np.clip(np.mean(y_train_binary), 1e-10, 1 - 1e-10) 
            self.initial_prediction = np.log(p_initial / (1 - p_initial)) # Log-odds

        current_raw_predictions_train = np.full(y_train.shape[0], self.initial_prediction)
        
        # Initialize raw predictions for validation set if exists
        if X_val is not None:
            current_raw_predictions_val = np.full(y_val.shape[0], self.initial_prediction)
            early_stopper = EarlyStopping(patience=self.early_stopping_rounds, min_delta=self.min_delta)
        else:
            current_raw_predictions_val = None
            early_stopper = None # No early stopping if no validation set

        # Boosting loop
        for i in range(self.n_estimators):
            # 2. Compute gradients (using training data)
            gradients = -self.loss_fn.gradient(y_train, current_raw_predictions_train)
            
            # 3. Fit a decision tree to the gradients
            tree_learner = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                tree_type='regressor', # Trees in GBM always predict continuous values
                feature_types=self.feature_types
            )
            tree_learner.fit(X_train, gradients) # Fit to the negative gradients from training data

            # 4. Get predictions from the new tree
            tree_predictions_train = tree_learner.predict(X_train)
            
            # Get predictions for validation set too
            if X_val is not None:
                tree_predictions_val = tree_learner.predict(X_val)

            # 5. Add shrunken prediction to ensemble
            current_raw_predictions_train += self.learning_rate * tree_predictions_train
            self.estimators.append(tree_learner) # Store the trained tree

            # Calculate and store training loss
            current_train_loss = self.loss_fn.loss(y_train, current_raw_predictions_train)
            self.train_loss_history.append(current_train_loss)
            
            # Calculate and check validation loss for early stopping
            if X_val is not None:
                current_raw_predictions_val += self.learning_rate * tree_predictions_val
                current_val_loss = self.loss_fn.loss(y_val, current_raw_predictions_val)
                self.val_loss_history.append(current_val_loss)

                if early_stopper(current_val_loss):
                    print(f"Early stopping at round {i+1}/{self.n_estimators}. Validation loss did not improve for {self.early_stopping_rounds} rounds.")
                    break 
            
            # Accumulate feature importances from the current tree
            # tree_learner.feature_importances_ is accumulated in tree.py during _find_best_split
            if tree_learner.feature_importances_ is not None:
                self.feature_importances_ += tree_learner.feature_importances_

        # Normalize feature importances after the boosting loop
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0: # Avoid division by zero if all importances are 0
            self.feature_importances_ /= total_importance
        else:
            self.feature_importances_ = np.zeros(n_features) # All features are equally unimportant (or no splits occurred)

    def predict(self, X):
        # Sum predictions from all estimators
        # Start with the initial prediction F0
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for tree_learner in self.estimators:
            predictions += self.learning_rate * tree_learner.predict(X)
        
        # For classification, apply sigmoid to convert raw scores to probabilities
        if self.loss_type == 'classifier':
            return sigmoid(predictions)
        else: # For regression, return raw sum
            return predictions