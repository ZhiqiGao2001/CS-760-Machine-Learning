import os
import re
import numpy as np
from functools import reduce
from operator import itemgetter
import random
from torchvision import datasets, transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# read files
def load_data(filepath: str) -> str:
    with open(filepath, 'r') as f:
        lines = f.read()
    lines = re.sub('[^a-zA-Z ]', '', lines)
    return lines


all_file_name = []
dataset_path = "languageID"

for lang in ['e', 'j', 's']:
    for i in range(20):
        file_path = os.path.join(dataset_path, lang + str(i) + '.txt')
        all_file_name.append(file_path)

N = len(all_file_name)
labels = ["e"] * 20 + ["j"] * 20 + ["s"] * 20
all_words = []

for file_path in all_file_name:
    word_file = load_data(file_path)
    all_words.append(word_file)

print("Question 3.1")
training = [i for i in range(0, 10)] + [i for i in range(20, 30)] + [i for i in range(40, 50)]
training_size = len(training)
# Set parameters
alpha = 1/2
label_count = np.zeros(3) # [en, jp, sp]
langs = np.array(["e", "j", "s"])
L = len(label_count)

# Update label counts
for label in [labels[i] for i in training]:
    label_count[np.where(langs == label)[0][0]] += 1

# Initialize priors
prior = np.full(L, 1/L)

# Update priors using label counts
prior = (label_count + alpha) / (training_size + L*alpha)

print(prior)

print("\nQuestion 3.2, 3.3")
# Define an alphabet list
alphabet = list("abcdefghijklmnopqrstuvwxyz ")
class_conditional_probability = []

for l in range(3):
    specific_language = [0] * 27  # Initialize ccpl with 27 zeros
    for i in range(training_size):
        if labels[training[i]] == langs[l]:
            for j in range(len(all_words[training[i]])):
                letter_index = alphabet.index((all_words[training[i]][j]))
                specific_language[letter_index] += 1

    denom = sum(specific_language) + alpha * 27
    # Update ccp list
    specific_language = [(count + alpha) / denom for count in specific_language]

    class_conditional_probability.append(specific_language)

# Print results
for i in range(3):
    print(class_conditional_probability[i])

print("\nQuestion 3.4")
counts = [0] * 27
# read from e10.txt
for word in all_words[10]:
    counts[alphabet.index(word)] += 1
print(counts)

print("\nQuestion 3.5")
log_prob = []
for i in range(3):
    logprob = sum(counts[x] * np.log(class_conditional_probability[i][x]) for x in range(27))
    log_prob.append(logprob)
print(log_prob)

print("\nQuestion 3.6")
# print the language with the highest posterior probability
print("The language with the highest posterior probability is: ", langs[np.argmax(log_prob)])

print("\nQuestion 3.7")
tests = [i for i in range(10, 20)] + [i for i in range(30, 40)] + [i for i in range(50, 60)]
test_size = len(tests)
confusion = np.zeros((3, 3))

for i in range(test_size):
    counts = [0] * 27
    for word in all_words[tests[i]]:
        counts[alphabet.index(word)] += 1

    log_prob = []
    for j in range(3):
        logprob = sum(counts[x] * np.log(class_conditional_probability[j][x]) for x in range(27))
        log_prob.append(logprob)
    true_label = np.where(langs == labels[tests[i]])[0][0]
    confusion[true_label][np.argmax(log_prob)] += 1

print(confusion)


########################################################################################################################
# Hyperparameters
d = 784  # Input dimension (MNIST images are 28x28 = 784)
d1 = 300
k = 10  # Output dimension (MNIST has 10 classes)

learning_rate = 0.035
batch_size = 64
num_epochs = 35

# Add a transformation to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST datasets with transformations
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Loaders with transformed data
train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class NeuralNetwork_Numpy:
    def __init__(self, d, d1, k):
        self.weights1 = np.random.uniform(-1, 1, (d, d1))
        self.weights2 = np.random.uniform(-1, 1, (d1, k))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2)
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, x, y):
        m = x.shape[0]
        dz2 = self.a2 - np.eye(self.a2.shape[1])[y]  # Convert labels to one-hot encoding
        dw2 = np.dot(self.a1.T, dz2) / m

        dz1 = np.dot(dz2, self.weights2.T) * self.a1 * (1 - self.a1)
        dw1 = np.dot(x.T, dz1) / m
        return dw1, dw2

    def update_parameters(self, dw1, dw2, learning_rate):
        self.weights1 -= learning_rate * dw1
        self.weights2 -= learning_rate * dw2

# Initialize the model, loss function, and optimizer
model = NeuralNetwork_Numpy(d, d1, k)

# Define the CrossEntropyLoss function
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

# Training loop
train_losses = []  # List to store the training losses

# Lists to store test error and epochs
epochs_list = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    total_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.view(-1, d).numpy()  # Flatten the input images
        labels = labels.numpy()

        # Forward pass
        outputs = model.forward(images)
        loss = cross_entropy_loss(outputs, labels)

        # Backward pass
        dw1, dw2 = model.backward(images, labels)

        # Update parameters
        model.update_parameters(dw1, dw2, learning_rate)

        total_loss += loss

        predicted = np.argmax(outputs, axis=1)
        total_train += labels.shape[0]
        correct_train += np.sum(predicted == labels)

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Calculate test accuracy
    correct_test = 0
    total_test = 0

    for images, labels in test_loader:
        images = images.view(-1, d).numpy()
        labels = labels.numpy()
        outputs = model.forward(images)
        predicted = np.argmax(outputs, axis=1)
        total_test += labels.shape[0]
        correct_test += np.sum(predicted == labels)

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}')
    epochs_list.append(epoch + 1)

# print the final epoch's test accuracy
print(f'Final test accuracy: {test_accuracy}')
print(f'Final test error rate: {100 - test_accuracy}')

plt.plot(epochs_list, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs_list, test_accuracies, label='Test Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracy')
plt.legend()
plt.show()

class NeuralNetwork(nn.Module):
    def __init__(self, d, d1, k):
        super(NeuralNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(d, d1),
            nn.Sigmoid(),
            nn.Linear(d1, k),
            # nn.Softmax()
        )

        # # Initialize the weights to zero as required
        # for layer in self.features:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.zeros_(layer.weight)

        # Initialize all linear layer weights randomly between -1 and 1
        # for layer in self.features:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.uniform_(layer.weight, -1, 1)

    def forward(self, x):
        x = self.features(x)
        return x


# Initialize the model, loss function, and optimizer
model = NeuralNetwork(d, d1, k)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []  # List to store the training losses

# Lists to store test error and epochs
epochs_list = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    total_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.view(-1, d))  # Flatten the input images
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Calculate test accuracy
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.view(-1, d))
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}')
    epochs_list.append(epoch + 1)


# print the final epoch's test accuracy
print(f'Final test accuracy: {test_accuracy}')
print(f'Final test error rate: {100 - test_accuracy}')


# Plotting both training and test accuracy
plt.plot(epochs_list, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs_list, test_accuracies, label='Test Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracy')
plt.legend()
plt.show()


