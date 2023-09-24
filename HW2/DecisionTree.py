import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, split_feature=None, threshold=None, left=None, right=None, label=None):
        self.split_feature = split_feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(y, y_left, y_right):
    H_parent = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    return H_parent - p_left * H_left - p_right * H_right

def find_best_split(X, y):
    best_gain = 0
    best_split = None
    for j in range(X.shape[1]):
        thresholds = np.unique(X[:, j])
        for c in thresholds:
            mask = X[:, j] >= c
            y_left = y[mask]
            y_right = y[~mask]
            if len(y_left) > 0 and len(y_right) > 0:
                gain = information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (j, c)
    return best_split

def build_tree(X, y):
    if len(set(y)) == 1:
        return TreeNode(label=y[0])
    if len(X) == 0:
        return TreeNode(label=1)  # Predict class 1 if no data
    split_feature, threshold = find_best_split(X, y)
    if split_feature is None:
        return TreeNode(label=1)  # Predict class 1 if no informative split
    mask = X[:, split_feature] >= threshold
    X_left, y_left = X[mask], y[mask]
    X_right, y_right = X[~mask], y[~mask]
    left_subtree = build_tree(X_left, y_left)
    right_subtree = build_tree(X_right, y_right)
    return TreeNode(split_feature=split_feature, threshold=threshold, left=left_subtree, right=right_subtree)

def predict(tree, x):
    if tree.label is not None:
        return tree.label
    if x[tree.split_feature] >= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)


def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            x1, x2, y = map(float, line.strip().split())
            data.append([x1, x2, y])
        return np.array(data)


def question_2_2():
    # Generate the training set
    data = np.array([[1.0, 2.0, 0],
                     [1.5, 2.5, 0],
                     [2.0, 3.0, 1],
                     [2.5, 3.5, 1],
                     [3.0, 4.0, 0]])

    X = data[:, :-1]
    y = data[:, -1]

    # Build the tree
    tree = build_tree(X, y)

    # Plot the training set
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', marker='x', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Class 1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Training Set')
    plt.show()

    return tree

def print_tree(node, depth=0):
    if node.label is not None:
        print(f"{'  ' * depth}Predict: Class {node.label}")
    else:
        print(f"{'  ' * depth}Split on feature {node.split_feature} at threshold {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

# Assuming you have already built the tree

# Example usage
if __name__ == '__main__':
    # Example usage


    # # Assuming your data is in the format x1 x2 y
    data = read_data('Homework 2 data/Druns.txt')

    X = data[:, :-1]
    y = data[:, -1]

    # Build the tree
    tree = build_tree(X, y)
    print_tree(tree)

    # # Example prediction
    # new_data_point = np.array([1.493761, -0.753345])
    # prediction = predict(tree, new_data_point)
    # print(f"Predicted class: {prediction}")
