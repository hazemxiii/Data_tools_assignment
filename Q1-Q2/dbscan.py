from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X = np.array([[1,1],
[1,2],
[1,3],
[2,2],
[3,3],
[3,4],
[4,3],
[4,4]
])

# Fit the DBSCAN algorithm
dbscan = DBSCAN(eps=2**.5, min_samples=1)
labels = dbscan.fit_predict(X)

# print(labels)
# exit()

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title('DBSCAN Clustering')
plt.show()