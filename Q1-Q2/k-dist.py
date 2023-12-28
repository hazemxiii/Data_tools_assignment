import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd

# Generate sample data (replace this with your dataset)
X = np.array([[1,1],
[1,2],
[1,3],
[2,2],
[3,3],
[3,4],
[4,3],
[4,4]])
l = 'p1 p2 p3 p4 p5 p6 p7 p8'.split()
d = (pd.DataFrame(cdist(X,X,metric='euclidean'),columns=l,index = l))
print(d[d['p1']<=1.42])
exit()

# Function to calculate k-distance for each data point
def calculate_k_distance(X, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    return np.sort(distances[:, -1])

# Function to plot the k-distance graph
def plot_k_distance_graph(k_distances):
    plt.plot(range(1, len(k_distances) + 1), k_distances, marker='o')
    plt.title('K-Distance Graph')
    plt.xlabel('Data Points (sorted)')
    plt.ylabel('K-Distance')
    plt.show()

# Function to find the knee point in the k-distance graph
def find_knee_point(k_distances):
    differences = np.diff(k_distances, 2)
    knee_point_index = np.argmax(differences) + 2  # Add 2 because of the double differentiation
    return knee_point_index

# Calculate k-distances
k_value = 4  # You can adjust the value of k
k_distances = calculate_k_distance(X, k_value)

# Plot the k-distance graph
plot_k_distance_graph(k_distances)

# Find the knee point and corresponding epsilon
knee_point_index = find_knee_point(k_distances)
epsilon = k_distances[knee_point_index - 1]  # Subtract 1 to get the correct index

print(f'Optimal Epsilon (knee point): {epsilon}')