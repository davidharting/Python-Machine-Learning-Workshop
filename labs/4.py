import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

print('C L U S T E R I N G\n\n')
np.random.seed(42)

# Load the data

policies = pd.read_csv('data/Rates.csv')

# Explore the Data
print(policies.head())

pd.plotting.scatter_matrix(
    frame = policies,
    alpha = 1,
    s = 100,
    diagonal = 'none'
)
# plt.show()
plt.close()

# Transform the data

X = policies.iloc[:, policies.columns != 'State']
X.Gender = X.Gender.apply(lambda x: 0 if x == "Female" else 1)
print(X.head())

# Cluster with k-Means
k_model = KMeans(n_clusters = 3, n_init = 10)
k_model.fit(X)
palette = { 0: '#fb8072', 1: '#80b1d3', 2: '#b3de69' }
k_colors = pd.Series(k_model.labels_).apply(lambda x:palette[x])
pd.plotting.scatter_matrix(
    frame = policies,
    color = k_colors,
    alpha = 1,
    s = 100,
    diagonal = 'none'
)
# plt.show()
plt.close()

plt.scatter(
    x = policies.Age,
    y = policies.BMI,
    color = k_colors)
plt.scatter(
    x = k_model.cluster_centers_[:,5],
    y = k_model.cluster_centers_[:,4],
    marker = 'x',
    color = "black",
    s = 100)
plt.xlabel = "BMI"
plt.ylabel = "Age"
# plt.show()
plt.close()

# Cluster with Hierarchical Clustering
h_model = AgglomerativeClustering(n_clusters = 3)
h_model.fit(X)
# plot dendrogram
children = h_model.children_
distance = np.arange(children.shape[0])
observations = np.arange(2, children.shape[0] + 2)
linkage_matrix = np.column_stack([children, distance, observations]).astype(float)
dendrogram(
  Z = linkage_matrix,
  leaf_font_size = 8,
  color_threshold = 1939
)

# Map previous three colors to each cluster
h_colors = pd.Series(h_model.labels_).apply(lambda x: palette[x])
plt.scatter(
    x = policies.Age,
    y = policies.BMI,
    color = h_colors)
plt.xlabel = "Age"
plt.ylabel = "BMI"
plt.show()
plt.close()
