#%% Demo 4 - Clustering

#%% Load the OS library
import os

#%% Set the working directory
os.chdir("C:\\Users\\Matthew\\Dropbox\\Professional\\Workshops\\Practical Machine Learning with Python\\Data")

#%% Load the pandas library
import pandas as pd

#%% Read the Iris CSV file
iris = pd.read_csv("Iris.csv")

#%% Create features (X)
X = iris.iloc[:, 0:4]

#%% Inspect features (X)
X.head()

#%% Import numpy library
import numpy as np

#%% Set random number seed
np.random.seed(42)

#%% Cluster with k-Means

#%% Import k-means from sklearn
from sklearn.cluster import KMeans

#%% Create a k-means model
k_model = KMeans(
    n_clusters = 3,
    n_init = 10)

#%% Fit the model
k_model.fit(X)

#%% Load the matplotlib library
import matplotlib.pyplot as plt

#%% Create a safe color palette
palette = {0:'#fb8072', 1:'#80b1d3', 2:'#b3de69'}

#%% Create colors for each cluster
# TODO: NEED TO GET THIS TO WORK
k_colors = pd.Series(k_model.labels_) \
    .apply(lambda x:palette[x])
          
#%% Plot clusters
plt.scatter(
    x = iris.Petal_Length,
    y = iris.Petal_Width,
    color = k_colors)

plt.xlabel = "Petal Length"
plt.ylabel = "Petal Width"

#%% Plot centroids
plt.scatter(
    x = k_model.cluster_centers_[:,2],
    y = k_model.cluster_centers_[:,3],
    marker = 'x',
    color = "black",
    s = 100)

plt.show()

#%% Cluster with Hierarchical Clustering

#%% Import agglomerative clustering from sklearn
from sklearn.cluster import AgglomerativeClustering

#%% Create a hierarchical cluster model
h_model = AgglomerativeClustering(
    n_clusters = 3,
    affinity = "euclidean", 
    linkage = "ward")

#%% Fit the model
h_model.fit(X)

#%% Import dendrogram
from scipy.cluster.hierarchy import dendrogram

#%% Plot the dendrogram
children = h_model.children_

distance = np.arange(children.shape[0])

observations = np.arange(2, children.shape[0] + 2)

linkage_matrix = np.column_stack([children, distance, observations]).astype(float)

dendrogram(
    Z = linkage_matrix,
    leaf_font_size = 8,
    color_threshold = 147)

#%% Create colors for each cluster
h_colors = pd.Series(h_model.labels_) \
    .apply(lambda x:palette[x])

#%% Plot clusters
plt.scatter(
    x = iris.Petal_Length,
    y = iris.Petal_Width,
    color = h_colors)

plt.xlabel = "Petal Length"
plt.ylabel = "Petal Width"
plt.show()
