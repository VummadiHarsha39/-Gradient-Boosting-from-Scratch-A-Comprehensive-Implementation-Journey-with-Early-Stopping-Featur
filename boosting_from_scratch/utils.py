# boosting_from_scratch/utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap # For plotting decision boundaries
# You might also need to import your DecisionTree and Node classes for type hints
# from .tree import DecisionTree, Node # Commented out for now to avoid circular imports if not needed at runtime

def plot_tree_decision_boundary(tree_model, X, y, feature_names=None, ax=None):
    """
    Plots the decision boundary of a 2D DecisionTree classifier.
    Assumes tree_model is a DecisionTree (classifier) trained on 2 numerical features.
    """
    if tree_model.tree_type != 'classifier':
        print("Warning: Plotting decision boundary is best for classifiers.")
        return

    if X.shape[1] != 2:
        print("Warning: Decision boundary plotting for trees only supports 2 features currently.")
        return

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a meshgrid to evaluate the model across the feature space
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict class for each point in the meshgrid
    Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) # For scatter points

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the decision boundary
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
               edgecolor='k', s=20)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title("Decision Tree Classifier Decision Boundary")
    ax.set_xlabel(feature_names[0] if feature_names else "Feature 0")
    ax.set_ylabel(feature_names[1] if feature_names else "Feature 1")
    ax.grid(True, linestyle='--', alpha=0.6)

# --- Helper to extract splits for drawing (Advanced, will implement in DecisionTree class instead) ---
# We will draw explicit lines on top of this decision boundary if we want to show splits.
# The actual lines are easier drawn in demo.ipynb by inspecting tree_model.tree.