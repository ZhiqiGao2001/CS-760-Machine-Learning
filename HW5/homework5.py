import numpy as np
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


########################################################################################################################
# Problem 1
def generate_synthetic_dataset(sigma):
    # Define the mean and covariance matrices for each distribution
    mean_a = np.array([-1, -1])
    cov_a = sigma * np.array([[2, 0.5], [0.5, 1]])

    mean_b = np.array([1, -1])
    cov_b = sigma * np.array([[1, -0.5], [-0.5, 2]])

    mean_c = np.array([0, 1])
    cov_c = sigma * np.array([[1, 0], [0, 2]])

    # Sample points from each distribution
    points_a = np.random.multivariate_normal(mean_a, cov_a, 100)
    points_b = np.random.multivariate_normal(mean_b, cov_b, 100)
    points_c = np.random.multivariate_normal(mean_c, cov_c, 100)

    # Concatenate the points to form the final dataset
    synthetic_dataset = np.vstack([points_a, points_b, points_c])

    return synthetic_dataset


# Define a list of sigma values
sigma_values = [0.5, 1, 2, 4, 8]

# Generate datasets for each sigma value
datasets = [generate_synthetic_dataset(sigma) for sigma in sigma_values]


def kmeans(X, k, max_iters=30, num_restarts=100):
    best_centers = None
    best_labels = None
    best_cost = float('inf')

    for _ in range(num_restarts):
        # Initialize cluster centers randomly
        centers = X[np.random.choice(len(X), k, replace=False)]

        for _ in range(max_iters):
            # Assign each point to the nearest center
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

            # Update cluster centers
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])

            if np.all(new_centers == centers):
                break

            centers = new_centers

        # Calculate the cost (sum of squared distances)
        cost = np.sum((X - centers[labels]) ** 2)

        if cost < best_cost:
            best_centers = centers
            best_labels = labels
            best_cost = cost

    return best_centers, best_labels, best_cost


def calculate_accuracy(labels):
    # Assign cluster labels to true distribution labels based on majority vote
    cluster_to_distribution = {}

    for cluster in range(k):
        cluster_indices = np.where(labels == cluster)[0]
        distribution_counts = [0, 0, 0]

        for idx in cluster_indices:
            if idx < 100:
                distribution_counts[0] += 1
            elif idx < 200:
                distribution_counts[1] += 1
            else:
                distribution_counts[2] += 1

        cluster_to_distribution[cluster] = np.argmax(distribution_counts)

    # Map cluster labels to true distribution labels
    predicted_labels = np.array([cluster_to_distribution[label] for label in labels])

    # Calculate accuracy
    correct = np.sum(predicted_labels == true_labels)
    total = len(true_labels)
    accuracy = correct / total

    return accuracy


# Define a list of sigma values
sigma_values = [0.5, 1, 2, 4, 8]

# Define the number of clusters (k)
k = 3

# Initialize lists to store results
custom_kmeans_results = []

for i in range(5):
    custom_kmeans_centers, custom_kmeans_labels, custom_kmeans_cost = kmeans(datasets[i], k)
    custom_kmeans_results.append((custom_kmeans_centers, custom_kmeans_labels, custom_kmeans_cost))


true_labels = np.hstack([np.zeros(100), np.ones(100), np.full(100, 2)])

for i, sigma in enumerate(sigma_values):
    print('Sigma = {}'.format(sigma))
    print('Custom K-means cost = {}'.format(custom_kmeans_results[i][2]))

    # Custom K-means
    custom_kmeans_labels = custom_kmeans_results[i][1]
    custom_accuracy = calculate_accuracy(custom_kmeans_labels)
    print('Custom K-means accuracy = {:.2f}%'.format(custom_accuracy * 100))


def compute_accuracy_and_objective(dataset, true_labels):
    gmm = GaussianMixture(n_components=3)
    gmm.fit(dataset)
    predicted_labels = gmm.predict(dataset)
    accuracy = np.sum(predicted_labels == true_labels)/300
    objective = -gmm.score(dataset)  # Get the negative log-likelihood
    return accuracy, objective

accuracy_gmm = []
objective_gmm = []

# Iterate over each sigma value
for i in range(5):
    best_accuracy = 0
    best_objective = float('inf')
    for _ in range(1):
        accuracy, objective = compute_accuracy_and_objective(datasets[i], true_labels)
        if objective < best_objective:  # Find the lowest objective value
            best_accuracy = accuracy
            best_objective = objective

    accuracy_gmm.append(best_accuracy)
    objective_gmm.append(best_objective)

for i, sigma in enumerate(sigma_values):
    print('Sigma = {}'.format(sigma))
    print('GMM lowest objective value = {:.2f}'.format(objective_gmm[i]))
    print('GMM accuracy = {:.2f}%'.format(accuracy_gmm[i] * 100))


########################################################################################################################
