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
    best_split = (None, None)
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
    if len(X) == 0:
        return TreeNode(label=1)  # Predict class 1 if no data
    if len(set(y)) == 1:
        return TreeNode(label=y[0])  # Predict class if only one class
    split_feature, threshold = find_best_split(X, y)  # Find best split
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
    data = np.array([[1, 1, 0],
                     [2, 2, 0],
                     [1, 2, 1],
                     [2, 1, 1]])

    X = data[:, :-1]
    y = data[:, -1]

    # Build the tree
    tree = build_tree(X, y)

    # Plot the training set
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', marker='x', label='Label 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Label 1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Training Set')
    plt.show()
    return tree


def question_2_3(X, y):
    info = []
    for j in range(X.shape[1]):
        thresholds = np.unique(X[:, j])
        for c in thresholds:
            mask = X[:, j] >= c
            y_left = y[mask]
            y_right = y[~mask]
            gain = information_gain(y, y_left, y_right)
            info.append((j, c, gain))
    return info


def convert_tree_to_rules(node, parent_condition=""):
    if node.label is not None:
        print(f"If {parent_condition} then Label {node.label}")
    else:
        left_condition = f"{parent_condition} and x{node.split_feature + 1} >= {node.threshold}"
        right_condition = f"{parent_condition} and x{node.split_feature + 1} < {node.threshold}"
        convert_tree_to_rules(node.left, left_condition)  # Removing the leading ' and '
        convert_tree_to_rules(node.right, right_condition)  # Removing the leading ' and '


def scatter_plot_from_dataset(data):
    # Separate the coordinates and labels
    X = data[:, :-1]
    y = data[:, -1]

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', marker='x', s=5, label='Label 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', s=5, label='Label 1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Scatter Plot')
    plt.show()


def draw_uniform_points(tree, x1_interval, x2_interval, num_points):
    x1_values = np.linspace(x1_interval[0], x1_interval[1], num_points)
    x2_values = np.linspace(x2_interval[0], x2_interval[1], num_points)

    X_test = np.array([[x1, x2] for x1 in x1_values for x2 in x2_values])
    y_pred = [int(predict(tree, x)) for x in X_test]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='plasma', marker='.')
    plt.xlim(x1_interval)
    plt.ylim(x2_interval)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary')
    plt.show()



if __name__ == '__main__':
    print()
    # question_2_2
    # tree = question_2_2()
    # print(tree.left, tree.right, tree.split_feature, tree.threshold, tree.label)

    # question_2_3
    # data = read_data('Homework 2 data/Druns.txt')
    # X = data[:, :-1]
    # y = data[:, -1]
    # info = question_2_3(X, y)
    # for i,j,k in info:
    #     print(f"For candidate cut x{i+1} >= {j}, the information gain ratio is {k}")


    # question_2_4
    # data = read_data('Homework 2 data/D3leaves.txt')
    # X = data[:, :-1]
    # y = data[:, -1]
    # tree = build_tree(X, y)
    # print("Rules:")
    # convert_tree_to_rules(tree)

    # question_2_5
    # data = read_data('Homework 2 data/D1.txt')
    # X = data[:, :-1]
    # y = data[:, -1]
    # tree = build_tree(X, y)
    # print("Rules:")
    # convert_tree_to_rules(tree)
    # data = read_data('Homework 2 data/D2.txt')
    # X = data[:, :-1]
    # y = data[:, -1]
    # tree = build_tree(X, y)
    # print("Rules:")
    # convert_tree_to_rules(tree)

    # question_2_6
    # data = read_data('Homework 2 data/D1.txt')
    # scatter_plot_from_dataset(data)
    # data = read_data('Homework 2 data/D2.txt')
    # scatter_plot_from_dataset(data)
    data = read_data('Homework 2 data/D1.txt')
    X = data[:, :-1]
    y = data[:, -1]
    tree = build_tree(X, y)
    x1_interval = (0, 1)
    x2_interval = (0, 1)
    num_points = 200
    draw_uniform_points(tree, x1_interval, x2_interval, num_points)
    data = read_data('Homework 2 data/D2.txt')
    X = data[:, :-1]
    y = data[:, -1]
    tree = build_tree(X, y)
    x1_interval = (0, 1)
    x2_interval = (0, 1)
    num_points = 200
    draw_uniform_points(tree, x1_interval, x2_interval, num_points)

    # data = read_data('Homework 2 data/Druns.txt')
    #
    # X = data[:, :-1]
    # y = data[:, -1]
    #
    # # Build the tree
    # tree = build_tree(X, y)
    # new_data_point = np.array([1.493761, -0.753345])
    # prediction = predict(tree, new_data_point)
    # print(f"Predicted class: {prediction}")
