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
    def __init__(self, max_depth=None, min_samples_split=2, tree_type='regressor'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_type = tree_type
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            leaf_value = self._calculate_leaf_value(y) 
            return Node(value=leaf_value, n_samples=n_samples, loss=self._calculate_loss(y))

        
        best_split = self._find_best_split(X, y) 

        
        if best_split['feature_idx'] is None:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value, n_samples=n_samples, loss=self._calculate_loss(y))

        feature_idx = best_split['feature_idx']
        threshold = best_split['threshold']

  
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold) 

       
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        
        return Node(feature_idx=feature_idx, threshold=threshold, left=left_child, right=right_child,
                    n_samples=n_samples, loss=self._calculate_loss(y))
    
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
        
    def _is_numerical(self, feature_column):
        return np.issubdtype(feature_column.dtype, np.number)


    def _calculate_split_gain(self, X, y, feature_idx, threshold):
       

        parent_loss = self._calculate_loss(y) 
       
        if self._is_numerical(X[:, feature_idx]):
            left_mask = X[:, feature_idx] <= threshold
            right_mask = X[:, feature_idx] > threshold
        else: 
            left_mask = X[:, feature_idx] == threshold
            right_mask = X[:, feature_idx] != threshold
            
        y_left = y[left_mask]
        y_right = y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return -np.inf 
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)

        
        child_loss = (weight_left * self._calculate_loss(y_left) +
                      weight_right * self._calculate_loss(y_right))

        return parent_loss - child_loss    


    def _split_data(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        return X_left, y_left, X_right, y_right

    
    def _find_best_split(self, X, y):
       
        best_split = {'feature_idx': None, 'threshold': None, 'gain': -np.inf}
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return best_split

        current_loss = self._calculate_loss(y)

        for feature_idx in range(n_features):
            feature_column = X[:, feature_idx]

            if self._is_numerical(feature_column):
                unique_values = np.unique(feature_column)
                for i in range(len(unique_values) - 1):
                    threshold = (unique_values[i] + unique_values[i+1]) / 2
                    gain = self._calculate_split_gain(X, y, feature_idx, threshold)
                    if gain > best_split['gain']:
                        best_split = {'feature_idx': feature_idx, 'threshold': threshold, 'gain': gain}

        return best_split
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
