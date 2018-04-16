# Demo 2 - Classification

#%% Import the pandas library
import pandas as pd

#%% Read the Iris CSV file
iris = pd.read_csv("data/Iris.csv")

#%% Inspect the iris data set
iris.head()

#%% Import the matplotlib library
import matplotlib.pyplot as plt

#%% Create a safe color palette
palette = {
    'setosa':'#fb8072', 
    'versicolor':'#80b1d3', 
    'virginica':'#b3de69'}

#%% Map the colors to each species of iris flower
colors = iris.Species.apply(lambda x:palette[x])

#%% Create a scatterplot matrix
pd.plotting.scatter_matrix(
    frame = iris,
    color = colors,
    alpha = 1,
    s = 100,
    diagonal = "none")

#%% Create a scatterplot of petal length vs. width
plt.scatter(
    x = iris.Petal_Length,
    y = iris.Petal_Width,
    color = colors)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

#%% Create training and test set

#%% Create features (X)
X = iris.iloc[:, 0:4]

#%% Inspect features (X)
X.head()

#%% Create labels (y)
y = iris.Species

#%% Inspect labels
y.head()

#%% Import numpy library
import numpy as np

#%% Set random number seed
np.random.seed(123)

#%% Import train_test_split
from sklearn.model_selection import train_test_split

#%% Randomly sample 100 of 150 row indexes
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    train_size = 0.67)

#%% Inspect training and test sets
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test:  ", X_test.shape)
print("y_test:  ", y_test.shape)

#%% Predict with K-Nearest Neighbors

#%% Import neigbors from sklearn
from sklearn.neighbors import KNeighborsClassifier

#%% Create k-nearest neighbors model
knn_model = KNeighborsClassifier(
    n_neighbors = 3)

#%% Fit the model to the data
knn_model.fit(
	X = X_train, 
	y = y_train)

#%% Predict with the model
knn_predictions = knn_model.predict(X_test)

#%% Create confusion matrix
pd.crosstab(
    y_test, 
    knn_predictions, 
    rownames = ['Reference'], 
    colnames = ['Predicted'])

#%% Import accuracy_score 
from sklearn.metrics import accuracy_score

#%% Get prediction accuracy
knn_score = accuracy_score(
    y_true = y_test,
    y_pred = knn_predictions)

#%% Inspect results
print(knn_score)

#%% Visualize knn prediction results
plt.scatter(
    x = X_test.Petal_Length,
    y = X_test.Petal_Width,
    color = np.where(
        y_test == knn_predictions, 
        'black', 
        'red'))
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

#%% TODO: Visualize knn boundaries to explain
# Source: http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py

#%% Predict with Decision Tree Classifier

#%% Import decision tree classifier
from sklearn.tree import DecisionTreeClassifier

#%% Create a decision tree
tree_model = DecisionTreeClassifier(
    max_depth = 3)

#%% Fit the model
tree_model.fit(
    X = X_train, 
    y = y_train)

#%% Import tree visualizer
from sklearn.tree import export_graphviz

#%% Visualize the model
import graphviz
tree_graph = export_graphviz(
    decision_tree = tree_model,
    feature_names = list(X_train.columns.values),  
    class_names = list(y_train.unique()), 
    out_file = None) 
graphviz.Source(tree_graph) 

#%% TODO: Plot decision tree boundaries
# Source: http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html#sphx-glr-auto-examples-tree-plot-iris-py

#%% Predict with the model
tree_predictions = tree_model.predict(X_test)

#%% Get prediction accuracy
tree_score = accuracy_score(
    y_true = y_test, 
    y_pred = tree_predictions)

#%% Inspect results
print(tree_score)

#%% Visualize tree prediction results
plt.scatter(
    x = X_test.Petal_Length,
    y = X_test.Petal_Width,
    color = np.where(
        y_test == tree_predictions, 
        'black', 
        'red'))
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

#%% Predict with a Neural Network

#%% Import standard scaler
from sklearn.preprocessing import StandardScaler

#%% Create a standard scaler
scaler = StandardScaler()

#%% Fit the scaler to all training data
scaler.fit(X)

#%% Scale the training and test data
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

#%% Import the neural network classifier
from sklearn.neural_network import MLPClassifier

#%% Create a neural network classifier
neural_model = MLPClassifier(
    hidden_layer_sizes = (4),
    activation = "tanh",
    max_iter = 2000,
    verbose = True)

#%% TODO: Visualize neural network
# Source: http://viznet.readthedocs.io/en/latest/examples.html

#%% Train the model
neural_model.fit(
    X = X_train_scaled, 
    y = y_train)

#%% Predict with the model
neural_predictions = neural_model.predict(X_test_scaled)

#%% Get prediction accuracy
neural_score = accuracy_score(
    y_true = y_test, 
    y_pred = neural_predictions)

#%% Inspect results
print(neural_score)

#%% Compare results
print("KNN: ", knn_score)
print("Tree:", tree_score)
print("NNet:", neural_score)
