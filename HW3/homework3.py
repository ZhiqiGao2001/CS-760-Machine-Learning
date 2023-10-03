import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pandas as pd


# Iterate over all confidence values and calculate TPR and FPR for each value
def roc_curve_p1(y_true, confidence_scores):

    num_neg = y_true.count('-')
    num_pos = y_true.count('+')

    TP = 0
    FP = 0
    last_TP = 0
    fpr_points = [0]
    tpr_points = [0]

    for i in range(len(y_true)):
        if i > 0 and confidence_scores[i] != confidence_scores[i-1] and y_true[i] == '-' and TP > last_TP:
            FPR = FP / num_neg
            TPR = TP / num_pos
            fpr_points.append(FPR)
            tpr_points.append(TPR)
            last_TP = TP

        if y_true[i] == '+':
            TP += 1
        else:
            FP += 1

    FPR = FP / num_neg
    TPR = TP / num_pos
    fpr_points.append(FPR)
    tpr_points.append(TPR)

    return fpr_points, tpr_points


def plot_roc_curve(fpr_points, tpr_points):
    # Plot the ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_points, tpr_points, marker='o', linestyle='-', color='red', linewidth=4, markersize=12)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.show()


def custom_kNN_classifier(X_train, y_train, test_points, k):
    predicted_labels = []
    for test_point in test_points:
        distances = np.sqrt(np.sum((X_train - test_point)**2, axis=1))
        nearest_neighbor_indices = np.argpartition(distances, k)[:k]
        nearest_neighbor_labels = y_train[nearest_neighbor_indices]
        unique_labels, counts = np.unique(nearest_neighbor_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predicted_labels.append(predicted_label)
    return np.array(predicted_labels)


def custom_kNN_predict_proba(X_train, y_train, test_points, k):
    predicted_probs = []
    for test_point in test_points:
        distances = np.sqrt(np.sum((X_train - test_point)**2, axis=1))
        nearest_neighbor_indices = np.argpartition(distances, k)[:k]
        nearest_neighbor_labels = y_train[nearest_neighbor_indices]
        positive_count = np.sum(nearest_neighbor_labels)
        total_count = len(nearest_neighbor_labels)
        predicted_prob = positive_count / total_count
        predicted_probs.append(predicted_prob)
    return np.array(predicted_probs)


def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    return true_positives / (true_positives + false_negatives)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient

    return theta


def predict(X, theta):
    probabilities = sigmoid(np.dot(X, theta))
    return (probabilities >= 0.5).astype(int)


def logistic_regression_predict_proba(X, theta):
    probabilities = sigmoid(np.dot(X, theta))
    return probabilities


# # problem 1 - 5 (a)
# confidences = [0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1]
# labels = ['+', '+', '-', '+', '+', '-', '+', '+', '-', '-']
# fp, tp = roc_curve_p1(labels, confidences)
# print(np.column_stack((fp, tp)))
# plot_roc_curve(fp, tp)


# #  problem 2 - 1
# data = np.loadtxt('hw3Data/D2z.txt')
# X_train = data[:, :-1]
# y_train = data[:, -1]
#
# x_range = np.arange(-2, 2.1, 0.1)
# y_range = np.arange(-2, 2.1, 0.1)
# xx, yy = np.meshgrid(x_range, y_range)
# test_points = np.c_[xx.ravel(), yy.ravel()]
#
# predictions = custom_kNN_classifier(X_train, y_train, test_points, 1)
#
# plt.figure(figsize=(8, 8))
# plt.scatter(test_points[:, 0], test_points[:, 1], c=predictions, cmap='coolwarm', alpha=0.5, marker='.')
# plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', label='Class 0', marker='o')
# plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red', label='Class 1', marker='x')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('1-NN Classifier Predictions')
# plt.legend()
# plt.savefig('problem 2-1.png')
# plt.show()

file_path = 'hw3Data/emails.csv'
# Load the dataset
df = pd.read_csv(file_path)


X = df.iloc[:, 1:3001].values  # Features
y = df['Prediction'].values  # Labels
fold_indices = [(np.arange(1000), np.arange(1000, 5000)),
                (np.arange(1000, 2000), np.concatenate((np.arange(1000), np.arange(2000, 5000)))),
                (np.arange(2000, 3000), np.concatenate((np.arange(2000), np.arange(3000, 5000)))),
                (np.arange(3000, 4000), np.concatenate((np.arange(3000), np.arange(4000, 5000)))),
                (np.arange(4000, 5000), np.arange(4000))]

# problem 2 - 2
#
# for i, (test_idx, train_idx) in enumerate(fold_indices):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#
#     # Train and predict using 1-NN
#     y_pred = custom_kNN_classifier(X_train, y_train, X_test, 1)
#
#     # Calculate metrics
#     acc = np.mean(y_test == y_pred)
#     prec = precision(y_test, y_pred)
#     rec = recall(y_test, y_pred)
#
#     # Print results
#     print(f"Fold {i + 1}:")
#     print(f"  Accuracy: {acc:.4f}")
#     print(f"  Precision: {prec:.4f}")
#     print(f"  Recall: {rec:.4f}")


# #  problem 2 - 3
# num_iterations = 5000
# learning_rate = 0.0005
#
# for i, (test_idx, train_idx) in enumerate(fold_indices):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#
#     # Add bias term (intercept)
#     X_train = np.c_[np.ones(X_train.shape[0]), X_train]
#     X_test = np.c_[np.ones(X_test.shape[0]), X_test]
#
#     # Train logistic regression model
#     theta = logistic_regression(X_train, y_train, learning_rate, num_iterations)
#
#     # Predict using the trained model
#     y_pred = predict(X_test, theta)
#
#     # Calculate metrics
#     acc = np.mean(y_test == y_pred)
#     prec = precision(y_test, y_pred)
#     rec = recall(y_test, y_pred)
#
#     # Print results
#     print(f"Fold {i + 1}:")
#     print(f"  Accuracy: {acc:.4f}")
#     print(f"  Precision: {prec:.4f}")
#     print(f"  Recall: {rec:.4f}")


# #  problem 2 - 4
# # Define the list of k values to test
# k_values = [1, 3, 5, 7, 10]
#
# # Initialize a list to store average accuracies
# avg_accuracies = []
#
# # Perform 5-fold cross validation for each k
# for k in k_values:
#     accuracies = []
#
#     for test_idx, train_idx in fold_indices:
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#
#         # Predict
#         y_pred = custom_kNN_classifier(X_train, y_train, X_test, k)
#
#         # Calculate accuracy
#         accuracy = np.mean(y_test == y_pred)
#         accuracies.append(accuracy)
#
#     # Calculate average accuracy for this k
#     avg_accuracy = np.mean(accuracies)
#     avg_accuracies.append(avg_accuracy)
#     print(f'k = {k}, Average Accuracy = {avg_accuracy:.4f}')
#
# # Plot average accuracy versus k
# plt.plot(k_values, avg_accuracies, marker='o')
# plt.xlabel('k')
# plt.ylabel('Average Accuracy')
# plt.title('Average Accuracy vs. k')
# plt.grid(True)
# plt.savefig('problem 2-4.png')
# plt.show()


#  problem 2 - 5
train_idx = np.arange(4000)
test_idx = np.arange(4000, 5000)

# Split the data into training and test sets
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Train logistic regression
theta = logistic_regression(X_train, y_train, 0.0005, 5000)
lr_probs = logistic_regression_predict_proba(X_test, theta)

knn_probs = custom_kNN_predict_proba(X_train, y_train, X_test, 5)

# Calculate ROC curve for k-NN
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Calculate ROC curve for logistic regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Plot ROC curves
plt.figure(figsize=(10, 7))
plt.plot(fpr_knn, tpr_knn, color='navy', lw=3, label='k-NN (AUC = %0.2f)' % roc_auc_knn)
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=3, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('problem 2-5.png')
plt.show()
