import numpy as np

class Node:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None, n_samples=None, loss=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.n_samples = n_samples
        self.loss = loss

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, tree_type='regressor', feature_types=None):
        if tree_type not in ['regressor', 'classifier']:
            raise ValueError("tree_type must be 'regressor' or 'classifier'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_type = tree_type
        self.tree = None # Stores the root node of the trained tree
        self.feature_types = feature_types 

        # NEW: Initialize feature_importances_ for this single tree
        # This will accumulate gain from all splits within this tree.
        self.feature_importances_ = None 

    def fit(self, X, y):
        # Ensure feature_importances_ is initialized based on n_features for this tree's context
        # This ensures it's ready before _build_tree is called.
        self.feature_importances_ = np.zeros(X.shape[1])
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Stopping criteria for recursion:
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value, n_samples=n_samples, loss=self._calculate_loss(y))

        # Find the best split
        best_split = self._find_best_split(X, y)
        
        # If no good split is found (e.g., no gain)
        if best_split['feature_idx'] is None:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value, n_samples=n_samples, loss=self._calculate_loss(y))

        feature_idx = best_split['feature_idx']
        threshold = best_split['threshold']

        # Split the data
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)

        # Recursively build left and right subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        # Return a decision node with its children
        return Node(feature_idx=feature_idx, threshold=threshold, left=left_child, right=right_child,
                    n_samples=n_samples, loss=self._calculate_loss(y))

    def _find_best_split(self, X, y):
        # Finds the best feature and threshold to split the data.
        best_split = {'feature_idx': None, 'threshold': None, 'gain': -np.inf}
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            # No split possible, ensure feature_importances_ is correctly sized.
            # This case is less common for feature importance calc as it means no splits contribute.
            return best_split

        # Iterating through features and potential thresholds
        for feature_idx in range(n_features):
            feature_column = X[:, feature_idx]
            unique_values = np.unique(feature_column)
            is_numerical_feature = self._is_numerical(feature_idx)

            if is_numerical_feature:
                # NUMERICAL FEATURE SPLITTING
                for i in range(len(unique_values) - 1):
                    threshold = (unique_values[i] + unique_values[i+1]) / 2
                    gain = self._calculate_split_gain(X, y, feature_idx, threshold)
                    if gain > best_split['gain']:
                        best_split = {'feature_idx': feature_idx, 'threshold': threshold, 'gain': gain}
            else:
                # CATEGORICAL FEATURE SPLITTING
                for category_value in unique_values:
                    threshold = category_value 
                    gain = self._calculate_split_gain(X, y, feature_idx, threshold)
                    if gain > best_split['gain']:
                        best_split = {'feature_idx': feature_idx, 'threshold': threshold, 'gain': gain}
        
        # After finding the best split for *this node*, add its gain to the tree's feature importance
        # This gain represents how much this split contributes to reducing impurity in the tree.
        if best_split['feature_idx'] is not None: # Only if a valid split was found for this node
            self.feature_importances_[best_split['feature_idx']] += best_split['gain']

        return best_split

    def _split_data(self, X, y, feature_idx, threshold):
        # Splits the data into left and right subsets based on a feature and threshold.
        # Handles both numerical and categorical splits.
        
        feature_column = X[:, feature_idx]

        if self._is_numerical(feature_idx): # Pass index, not column
            left_mask = feature_column <= threshold
            right_mask = feature_column > threshold
        else:
            left_mask = feature_column == threshold
            right_mask = feature_column != threshold
            
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        return X_left, y_left, X_right, y_right

    def _calculate_split_gain(self, X, y, feature_idx, threshold):
        # Calculates the information gain (reduction in impurity) for a given split.
        parent_loss = self._calculate_loss(y)

        X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)

        if len(y_left) == 0 or len(y_right) == 0:
            return -np.inf # If a split results in an empty child, it's not a valid split (or infinite loss)

        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)

        child_loss = (weight_left * self._calculate_loss(y_left) +
                      weight_right * self._calculate_loss(y_right))

        return parent_loss - child_loss

    def _calculate_loss(self, y):
        if len(y) == 0:
            return 0
        
        if self.tree_type == 'regressor':
            return np.mean((y - np.mean(y))**2)
        elif self.tree_type == 'classifier':
            unique_classes = np.unique(y)
            gini = 0
            for cls in unique_classes:
                p_k = np.sum(y == cls) / len(y)
                gini += p_k**2
            return 1 - gini
        else:
            raise ValueError("Unknown tree_type. Must be 'regressor' or 'classifier'.")

    def _calculate_leaf_value(self, y):
        if len(y) == 0:
            return None
        
        if self.tree_type == 'regressor':
            return np.mean(y)
        elif self.tree_type == 'classifier':
            unique_classes, counts = np.unique(y, return_counts=True)
            return unique_classes[np.argmax(counts)]
        else:
            raise ValueError("Unknown tree_type. Must be 'regressor' or 'classifier'.")

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        feature_value = x[node.feature_idx]
        
        if self._is_numerical(node.feature_idx): # Pass the feature_idx from the node for lookup
            if feature_value <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else: # If not numerical, it's categorical
            if feature_value == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def _is_numerical(self, feature_idx): # Only accepts feature_idx
        if self.feature_types is None:
            raise ValueError("feature_types must be provided to DecisionTree when using mixed data for proper type inference.")
        
        return self.feature_types[feature_idx] == 'numerical'