import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import graphviz

def visualizeErrors(X_test, y_test, predictions):
  plt.scatter(
    x = X_test.Age,
    y = X_test.BMI,
    color = np.where(y_test == predictions, 'black', 'red')
  )
  plt.xlabel('Age')
  plt.ylabel('BMI')
  # plt.show()
  plt.close()

def predictWithKNearestNeighbors(X_train, X_test, y_train, y_test):
  print('\n\nK Nearest Neighbors')
  knn_model = KNeighborsClassifier(n_neighbors = 3)
  
  # Train the model
  knn_model.fit(X = X_train, y = y_train)

  # Test the model
  knn_predictions = knn_model.predict(X_test)

  # Confusion matrix displaying predictions
  confusion_matrix = pd.crosstab(
    y_test,
    knn_predictions,
    rownames = ['Referecence'],
    colnames = ['Predicted']
  )
  print(confusion_matrix)

  knn_score = accuracy_score(
    y_true = y_test,
    y_pred = knn_predictions
  )
  print('KNN score:', knn_score)

  # Visualize knn predictions (incorrect in red)
  plt.scatter(
    x = X_test.Age,
    y = X_test.BMI,
    color = np.where(y_test == knn_predictions, 'black', 'red')
  )
  plt.xlabel('Petal Length')
  plt.ylabel('Petal Width')
  # plt.show()
  plt.close()

  return knn_score

def decisionTreeClassifier(X_train, X_test, y_train, y_test):
  print('\n\nDecisionTreeClassifier')
  # Create model
  tree_model = DecisionTreeClassifier(max_depth = 3)
  
  # Train model
  tree_model.fit(X = X_train, y = y_train)
  print(tree_model)
  
  # Visualize model
  tree_graph = export_graphviz(
    decision_tree = tree_model,
    feature_names = list(X_train.columns.values),  
    class_names = list(y_train.unique()), 
    out_file = None) 
  graphviz.Source(tree_graph).render('output/2/decision_tree')
  
  # Test model
  tree_predictions = tree_model.predict(X_test)
  tree_score = accuracy_score(
    y_true = y_test,
    y_pred = tree_predictions
  )
  print('Tree Score:', tree_score)

  plt.scatter(
    x = X_test.Age,
    y= X_test.BMI,
    color = np.where(y_test == tree_predictions, 'black', 'red')
  )
  plt.xlabel('Age')
  plt.ylabel('BMI')
  # plt.show()
  plt.close()

  return tree_score

def neuralNetClassifier(X, X_train, X_test, y_train, y_test):
  print('\n\nNeural Net Classifier')
  
  # Scale the inputs
  scaler = StandardScaler()
  scaler.fit(X)
  print(scaler)

  x_train_scaled = scaler.transform(X_train)
  x_test_scaled = scaler.transform(X_test)

  # Create the model
  neural_model = MLPClassifier(
    hidden_layer_sizes = (4),
    activation = "tanh",
    max_iter = 2000
  )

  # Train
  neural_model.fit(
    X = x_train_scaled,
    y = y_train
  )
  print(neural_model)

  # Tets
  neural_predictions = neural_model.predict(x_test_scaled)

  # Score
  neural_score = accuracy_score(
    y_true = y_test,
    y_pred = neural_predictions
  )
  print('Nueral Net Score:', neural_score)

  visualizeErrors(X_test, y_test, neural_predictions)

  return neural_score
  
def main():
  policies = pd.read_csv("data/Risk.csv")
  print(policies.head(), '\n')

  palette = { 'Low':'#fb8072', 'High':'#80b1d3'}
  colors = policies.Risk.apply(lambda x:palette[x])

  scatter_matrix = pd.plotting.scatter_matrix(
    frame = policies,
    color = colors,
    alpha = 1,
    s = 100,
    diagonal = "none"
  )
  # plt.show() # Show the scatter_matrix
  plt.close()

  plt.scatter(
    x = policies.Age,
    y = policies.BMI,
    color = colors
  )
  plt.xlabel("Age")
  plt.ylabel("BMI")
  # plt.show() # This seems to be showing the scatter matrix. Not sure what's going on.
  plt.close()

  X = policies.loc[:, ["Age", "BMI", "Gender", "State.Rate"]]
  X.Gender = X.Gender.apply(lambda x: 0 if x == "Female" else 1)
  print(X.head())

  y = policies.Risk
  print(y.head())

  # Create training and test set
  np.random.seed(42)

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size = 0.8,
    test_size = 0.2
  )
  print("\nX_train: ", X_train.shape)
  print("y_train: ", y_train.shape)
  print("X_test:  ", X_test.shape)
  print("y_test:  ", y_test.shape, '\n')

  knn_score = predictWithKNearestNeighbors(X_train, X_test, y_train, y_test)
  tree_score = decisionTreeClassifier(X_train, X_test, y_train, y_test)
  neural_score = neuralNetClassifier(X, X_train, X_test, y_train, y_test)

  print('\n\n Scores')
  print("KNN: ", knn_score)
  print("Tree:", tree_score)
  print("NNet:", neural_score)

if __name__ == "__main__":
  main()
