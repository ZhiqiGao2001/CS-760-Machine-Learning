import matplotlib.pyplot as plt
import numpy as np

# Define the decision boundaries
def rule1(x1):
    return x1 >= 10.0

def rule2(x1, x2):
    return (x1 < 10.0) and (x2 >= 3.0)

def rule3(x1, x2):
    return (x1 < 10.0) and (x2 < 3.0)

# Generate a grid of points
x1_values = np.linspace(0, 20, 400)
x2_values = np.linspace(0, 10, 200)
X1, X2 = np.meshgrid(x1_values, x2_values)

# Apply rules to classify points
Z = np.zeros_like(X1)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if rule1(X1[i, j]):
            Z[i, j] = 1
        elif rule2(X1[i, j], X2[i, j]):
            Z[i, j] = 1
        elif rule3(X1[i, j], X2[i, j]):
            Z[i, j] = 0

# Create a filled contour plot
plt.contourf(X1, X2, Z, alpha=0.5, levels=[-1, 0, 1], colors=('red', 'blue'))

# Add labels
plt.xlabel('x1')
plt.ylabel('x2')

# Add legend
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Label 1'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Label 0')])

# Show the plot
plt.title('Decision Regions')
plt.show()

