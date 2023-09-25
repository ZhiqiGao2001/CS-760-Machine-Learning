import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
# Step 1: Generate points
np.random.seed(0)  # for reproducibility
a, b = 0, 2*np.pi  # Interval
n = 100  # Number of points
x_train = np.sort(np.random.uniform(a, b, n))

# Step 2: Compute y = sin(x)
y_train = np.sin(x_train)

# Step 3: Build Lagrange interpolation model
f = lagrange(x_train, y_train)

# Step 4: Generate test set
x_test = np.sort(np.random.uniform(a, b, n))
y_test = np.sin(x_test)

# calculate error for lagrange interpolation


# Step 5: Compute errors
train_error = np.mean((f(x_train) - y_train)**2)
test_error = np.mean((f(x_test) - y_test)**2)

print(f"Train Error: {train_error}")
print(f"Test Error: {test_error}")
print("=" * 30)
std_devs = [0.1, 0.5, 1.0]

for std_dev in std_devs:
    x_train_noisy = x_train + np.random.normal(0, std_dev, n)
    y_train_noisy = np.sin(x_train_noisy)

    f_noisy = lagrange(x_train_noisy, y_train_noisy)

    train_error_noisy = np.mean((f_noisy(x_train_noisy) - y_train_noisy) ** 2)
    test_error_noisy = np.mean((f_noisy(x_test) - y_test) ** 2)

    print(f"Standard Deviation: {std_dev}")
    print(f"Train Error (Noisy): {train_error_noisy}")
    print(f"Test Error (Noisy): {test_error_noisy}")
    print("=" * 30)




