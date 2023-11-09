import numpy as np
from scipy.stats import multivariate_normal
import warnings
import matplotlib.pyplot as plt
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

# Define the number of clusters (k)
k = 3

# Generate datasets for each sigma value
datasets = [generate_synthetic_dataset(sigma) for sigma in sigma_values]

true_labels = np.hstack([np.zeros(100), np.ones(100), np.full(100, 2)])


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


class CustomGaussianMixture:
    def __init__(self, n_components, max_iterations=100, tolerance=1e-4):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(self.n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(self.n_features) for _ in range(self.n_components)])

        for _ in range(self.max_iterations):
            # Expectation step
            responsibilities = self.calculate_responsibilities(X)

            # Maximization step
            self.update_parameters(X, responsibilities)
            if np.linalg.norm(self.old_means - self.means) < self.tolerance:
                break

    def calculate_responsibilities(self, X):
        responsibilities = np.zeros((self.n_samples, self.n_components))
        for i in range(self.n_components):
            responsibilities[:, i] = self.weights[i] * multivariate_normal.pdf(X, self.means[i], self.covariances[i])

        responsibilities /= np.sum(responsibilities, axis=1)[:, np.newaxis]
        return responsibilities

    def update_parameters(self, X, responsibilities):
        Nk = np.sum(responsibilities, axis=0)
        self.old_means = self.means.copy()
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        for i in range(self.n_components):
            diff = X - self.means[i]
            self.covariances[i] = np.dot((responsibilities[:, i] * diff.T), diff) / Nk[i]

        # Update weights
        self.weights = Nk / self.n_samples

    def predict(self, X):
        responsibilities = self.calculate_responsibilities(X)
        return np.argmax(responsibilities, axis=1)


def custom_neg_log_likelihood(gmm, dataset):
    n_samples, _ = dataset.shape
    responsibilities = gmm.calculate_responsibilities(dataset)
    likelihoods = np.zeros(n_samples)

    for i in range(gmm.n_components):
        likelihoods += responsibilities[:, i] * multivariate_normal.logpdf(dataset, gmm.means[i], gmm.covariances[i])

    return -np.sum(likelihoods)


def compute_accuracy_and_objective_custom(dataset):
    custom_gmm = CustomGaussianMixture(n_components=3)
    custom_gmm.fit(dataset)
    predicted_labels = custom_gmm.predict(dataset)
    objective = custom_neg_log_likelihood(custom_gmm, dataset)  # Get the negative log-likelihood
    return predicted_labels, objective

#
# # Initialize lists to store results
# custom_kmeans_results = []
# custom_kmeans_accuracy = []
#
#
# for i in range(5):
#     custom_kmeans_centers, custom_kmeans_labels, custom_kmeans_cost = kmeans(datasets[i], k)
#     custom_kmeans_results.append((custom_kmeans_centers, custom_kmeans_labels, custom_kmeans_cost))
#     custom_accuracy = calculate_accuracy(custom_kmeans_labels)
#     custom_kmeans_accuracy.append(custom_accuracy)
#
#
# for i, sigma in enumerate(sigma_values):
#     print('Sigma = {}'.format(sigma))
#     print('Custom K-means cost = {}'.format(custom_kmeans_results[i][2]))
#
#     # Custom K-means
#     print('Custom K-means accuracy = {:.2f}%'.format(custom_kmeans_accuracy[i] * 100))

# accuracy_gmm_custom = []
# objective_gmm_custom = []
#
# # Iterate over each sigma value
# for i in range(5):
#     best_label = 0
#     best_objective = float('inf')
#     best_label_custom = 0
#     best_objective_custom = float('inf')
#     for _ in range(10):
#         label_custom, objective_custom = compute_accuracy_and_objective_custom(datasets[i])
#         if objective_custom < best_objective_custom:  # Find the lowest objective value
#             best_label_custom = label_custom
#             best_objective_custom = objective_custom
#
#     accuracy_gmm_custom.append(calculate_accuracy(best_label_custom))
#     objective_gmm_custom.append(best_objective_custom)
#
# for i, sigma in enumerate(sigma_values):
#     print('Sigma = {}'.format(sigma))
#     print('Custom GMM lowest objective value = {:.2f}'.format(objective_gmm_custom[i]))
#     print('Custom GMM accuracy = {:.2f}%'.format(accuracy_gmm_custom[i] * 100))
#
#
# # Plot for objective values
# plt.figure(figsize=(10, 5))
# plt.plot(sigma_values, objective_gmm_custom, label='Custom GMM', marker='o')
# plt.plot(sigma_values, [result[2] for result in custom_kmeans_results], label='Custom K-means', marker='o')
# plt.xlabel('Sigma')
# plt.ylabel('Objective Value')
# plt.title('Objective Value vs. Sigma')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Plot for accuracy values
# plt.figure(figsize=(10, 5))
# plt.plot(sigma_values, accuracy_gmm_custom, label='Custom GMM', marker='o')
# plt.plot(sigma_values, custom_kmeans_accuracy, label='Custom K-means', marker='o')
# plt.xlabel('Sigma')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Sigma')
# plt.legend()
# plt.grid(True)
# plt.show()


########################################################################################################################
# Problem 2
def reconstruction_error(X, reconstructions):
    squared_diff = np.sum((X - reconstructions)**2)
    return squared_diff


def buggy_pca(X, d):
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    Z = np.dot(X, Vt[:d, :].T)
    reconstructions = np.dot(Z, Vt[:d, :])
    return reconstructions


def demeaned_pca(X, d):
    mean = np.mean(X, axis=0)
    X_demeaned = X - mean
    U, Sigma, Vt = np.linalg.svd(X_demeaned, full_matrices=False)
    Z = np.dot(X_demeaned, Vt[:d, :].T)
    reconstructions = np.dot(Z, Vt[:d, :]) + mean
    return reconstructions


def normalized_pca(X, d):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    U, Sigma, Vt = np.linalg.svd(X_normalized, full_matrices=False)
    Z = np.dot(X_normalized, Vt[:d, :].T)
    reconstructions = np.dot(Z, Vt[:d, :]) * std_dev + mean
    return reconstructions


def optimize_dimensionality_reduction(X, d):
    b = np.mean(X, axis=0)
    X_demeaned = X - b
    U, Sigma, Vt = np.linalg.svd(X_demeaned, full_matrices=False)
    Sigma_d = np.diag(Sigma.copy())
    reconstructed_data = np.dot(U[:, :d], np.dot(Sigma_d[:d, :d], Vt[:d, :])) + b
    return reconstructed_data


def plot_singular_values(X, start_idx=0, end_idx=None):
    mean = np.mean(X, axis=0)
    X_demeaned = X - mean
    U, Sigma, Vt = np.linalg.svd(X_demeaned, full_matrices=False)

    # Set default end_idx if not provided
    if end_idx is None:
        end_idx = len(Sigma)

    # Plot selected sigma values
    plt.plot(range(start_idx, end_idx), Sigma[start_idx:end_idx])
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title(f'Singular Values (Index {start_idx} to {end_idx-1})')
    plt.show()


data_2d = np.loadtxt('data/data2D.csv', delimiter=',')
data_1000d = np.loadtxt('data/data1000D.csv', delimiter=',')
d = 1

reconstructions_buggy = buggy_pca(data_2d, d)
reconstructions_demeaned = demeaned_pca(data_2d, d)
reconstructions_normalized = normalized_pca(data_2d, d)
reconstruction_dro = optimize_dimensionality_reduction(data_2d, d)

error_buggy = reconstruction_error(data_2d, reconstructions_buggy)
print(f"Reconstruction error (Buggy PCA) in data_2d: {error_buggy:.8f}")

error_demeaned = reconstruction_error(data_2d, reconstructions_demeaned)
print(f"Reconstruction error (Demeaned PCA) in data_2d: {error_demeaned:.8f}")

error_normalized = reconstruction_error(data_2d, reconstructions_normalized)
print(f"Reconstruction error (Normalized PCA) in data_2d: {error_normalized:.8f}")

error_dro = reconstruction_error(data_2d, reconstruction_dro)
print(f"Reconstruction error (Dimensionality Reduction Optimization) in data_2d: {error_dro:.8f}")


# plot_singular_values(data_1000d)
# plot_singular_values(data_1000d, 20, 40)

d_1000 = 30
error_buggy_1000d = reconstruction_error(data_1000d, buggy_pca(data_1000d, d_1000))
print(f"Reconstruction error (Buggy PCA) in data_1000d: {error_buggy_1000d:.8f}")

error_demeaned_1000d = reconstruction_error(data_1000d, demeaned_pca(data_1000d, d_1000))
print(f"Reconstruction error (Demeaned PCA) in data_1000d: {error_demeaned_1000d:.8f}")

error_normalized_1000d = reconstruction_error(data_1000d, normalized_pca(data_1000d, d_1000))
print(f"Reconstruction error (Normalized PCA) in data_1000d: {error_normalized_1000d:.8f}")

reconstruction_dro_1000= optimize_dimensionality_reduction(data_1000d, d_1000)
error_dro_1000d = reconstruction_error(data_1000d, reconstruction_dro_1000)
print(f"Reconstruction error (Dimensionality Reduction Optimization) in data_1000d: {error_dro_1000d:.8f}")


# plt.scatter(data_2d[:, 0], data_2d[:, 1], color='blue', label='Original Points', facecolors='none', edgecolors='b')
# plt.scatter(reconstructions_buggy[:, 0], reconstructions_buggy[:, 1], color='red', label='Reconstructed Points', marker='x')
# plt.legend()
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.title('Buggy PCA Reconstruction data2D.csv')
# plt.show()

# plt.scatter(data_2d[:, 0], data_2d[:, 1], color='blue', label='Original Points', facecolors='none', edgecolors='b')
# plt.scatter(reconstructions_demeaned[:, 0], reconstructions_demeaned[:, 1], color='red', label='Reconstructed Points', marker='x')
# plt.legend()
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.title('Demeaned PCA Reconstruction data2D.csv')
# plt.show()

# plt.scatter(data_2d[:, 0], data_2d[:, 1], color='blue', label='Original Points', facecolors='none', edgecolors='b')
# plt.scatter(reconstructions_normalized[:, 0], reconstructions_normalized[:, 1], color='red', label='Reconstructed Points', marker='x')
# plt.legend()
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.title('Normalized PCA Reconstruction data2D.csv')
# plt.show()


# plt.scatter(data_2d[:, 0], data_2d[:, 1], color='blue', label='Original Points', facecolors='none', edgecolors='b')
# plt.scatter(reconstruction_dro[:, 0], reconstruction_dro[:, 1], color='red', label='Reconstructed Points', marker='x')
# plt.legend()
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.title('Dimensionality Reduction Optimization Reconstruction data2D.csv')
# plt.show()

# d_values = np.arange(25, 35)
# errors = []
#
# for d in d_values:
#     reconstructed_data = optimize_dimensionality_reduction(data_1000d, d)
#     error = reconstruction_error(data_1000d, reconstructed_data)
#     errors.append(error)
#
# # Plotting the error vs d
# plt.plot(d_values, errors)
# plt.xlabel('d')
# plt.ylabel('Reconstruction Error')
# plt.title('Error vs Dimensionality (d)')
# plt.show()
