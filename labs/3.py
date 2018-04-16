import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def readData():
  print('\nread data')
  df = pd.read_csv('data/Rates.csv')
  print(df.head())
  return df

def explore(policies):
  print('\nExplore the data')

  pd.plotting.scatter_matrix(
    frame = policies,
    alpha = 1,
    s = 100,
    diagonal = 'none'
  )
  # plt.show()
  plt.close()

  correlations = policies.corr()
  print(correlations)

  sns.heatmap(
    data = correlations,
    cmap = sns.diverging_palette(
      h_neg = 10,
      h_pos = 220,
      as_cmap=True
    )
  )
  # plt.show()
  plt.close()

  age_rate_corr = policies.Age.corr(policies.Rate)
  print('\nCorrelation between Age and Rate', age_rate_corr)

  plt.scatter(
    x = policies.Age,
    y = policies.Rate
  )
  plt.xlabel('age')
  plt.ylabel('rate')
  # plt.show()
  plt.close()

def transform(policies):
  print('\n\nTransform')

  X = policies[['Gender', 'Age', 'State.Rate', 'BMI']]
  print(X.head())

  # one-hot encode Gender
  dummies = pd.get_dummies(X.Gender)
  print('Dummies', dummies.head())

  # Replace gender with dummy
  X = pd.concat([X, dummies], axis = 1)
  X = X.drop('Gender', 1)
  print('\nOne-hot-encoded gender:\n', X.head())

  y = policies.Rate
  print('y\n', y.head())

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size = 0.80,
    test_size = 0.20
  )

  print("X_train: ", X_train.shape)
  print("y_train: ", y_train.shape)
  print("X_test:  ", X_test.shape)
  print("y_test:  ", y_test.shape)

  return {
    "X": X,
    "X_train": X_train,
    "X_test": X_test,
    "y": y,
    "y_train": y_train,
    "y_test": y_test
  }

def simple(policies, X_train, X_test, y_train, y_test):
  print('\nSimple linear regression')
  simple_model = LinearRegression()
  x_train = X_train.loc[:, ['Age']]
  x_test = X_test.loc[:, ['Age']]

  simple_model.fit(
    X = x_train,
    y = y_train
  )
  print(simple_model)

  plt.scatter(
    x = policies.Age,
    y = policies.Rate,
    color = 'grey'
  )
  plt.plot(
    x_test,
    simple_model.predict(x_test),
    color = 'blue',
    linewidth = 3
  )
  plt.xlabel('Age')
  plt.ylabel('Rate')
  # plt.show()
  plt.close()

  print('y-intercept (b): ', simple_model.intercept_)
  print('slope (m)', simple_model.coef_[0])

  # Test
  simple_predictions = simple_model.predict(x_test)

  # Visualize prediction error
  # Plot the training set (grey dots)
  plt.scatter(
      x = x_train.Age,
      y = y_train,
      color = "grey",
      facecolor = "none")
  # Plot the predictions (blue dots)
  plt.scatter(
      x = x_test.Age,
      y = simple_predictions,
      color = "blue",
      marker = 'x')
  # Plot the correct answer (green dots)
  plt.scatter(
      x = x_test.Age,
      y = y_test,
      color = "green")
  # Plot the error (red lines)
  plt.plot(
      [x_test.Age, x_test.Age],
      [simple_predictions, y_test],
      color = "red",
      zorder = 0)
  # Finish the plot
  plt.xlabel("Age")
  plt.ylabel("Risk")
  # plt.show()
  plt.close()

  # Compute error (RMSE)
  simple_rmse = np.sqrt(
    np.mean(
      (y_test - simple_predictions) ** 2
    )
  )
  return simple_rmse

def multipleRegression(X_train, X_test, y_train, y_test):
  print('\n Multiple Regression')
  multiple_model = LinearRegression()
  multiple_model.fit(
    X = X_train,
    y = y_train
  )

  multiple_model.fit(
    X = X_train,
    y = y_train
  )

  print("{:<12}: {: .3f}".format("y-intercept", multiple_model.intercept_))
  for i, column_name in enumerate(X_train.columns):
    print("{:<12}: {: .3f}".format(
      column_name,
      multiple_model.coef_[i]
    ))
  
  multiple_predictions = multiple_model.predict(X_test)

  plt.scatter(
      x = X_train.Age,
      y = y_train,
      color = "black",
      facecolor = "none")
  plt.scatter(
      x = X_test.Age,
      y = multiple_predictions,
      color = "blue",
      marker = 'x')
  plt.scatter(
      x = X_test.Age,
      y = y_test,
      color = "green")
  plt.plot(
      [X_test.Age, X_test.Age],
      [multiple_predictions, y_test],
      color = "red",
      zorder = 0)
  plt.xlabel("Age")
  plt.ylabel("Rate")
  # plt.show()
  plt.close()

  multiple_rmse = np.sqrt(np.mean((y_test - multiple_predictions) ** 2))

  return multiple_rmse

def neuralNetwork(X, X_train, X_test, y, y_train, y_test):
  print('\nNeural Network')
  X_scaler = StandardScaler()
  y_scaler = StandardScaler()

  X_scaler.fit(X)
  y_scaler.fit(y.values.reshape(-1, 1))

  X_train_scaled = X_scaler.transform(X_train)
  X_test_scaled = X_scaler.transform(X_test)
  y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1))
  y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

  neural_model = MLPRegressor(
    hidden_layer_sizes = (4),
    activation = "tanh",
    solver = "lbfgs",
    max_iter = 1000
  )

  neural_model.fit(
    X = X_train_scaled,
    y = y_train_scaled.reshape(-1, ) # Why are we reshaping again?
  )

  scaled_predictions = neural_model.predict(X_test_scaled)
  neural_predictions = y_scaler.inverse_transform(scaled_predictions)

  plt.scatter(
    x = X_train.Age,
    y = y_train,
    color = "black",
    facecolor = "none")
  plt.scatter(
      x = X_test.Age,
      y = neural_predictions,
      color = "blue",
      marker = 'x')
  plt.scatter(
      x = X_test.Age,
      y = y_test,
      color = "green")
  plt.plot(
      [X_test.Age, X_test.Age],
      [neural_predictions, y_test],
      color = "red",
      zorder = 0)
  plt.xlabel("Age")
  plt.ylabel("Rate")
  plt.show()
  plt.close()

  neural_rmse = np.sqrt(np.mean((y_test - neural_predictions) ** 2))

  return neural_rmse

def main():
  print('Lab 3\n\n')

  np.random.seed(42) # Needed for test / train split sampling

  policies = readData()
  explore(policies)
  transformed = transform(policies)

  X = transformed['X']
  X_train = transformed['X_train']
  X_test = transformed['X_test']
  y = transformed['y']
  y_train = transformed['y_train']
  y_test = transformed['y_test']

  simple_rmse = simple(policies, X_train, X_test, y_train, y_test)
  multiple_rmse = multipleRegression(X_train, X_test, y_train, y_test)
  neural_rmse = neuralNetwork(X, X_train, X_test, y, y_train, y_test)

  print('\n\nAccuracy (rmse)')
  print('Simple:\t', simple_rmse)
  print('Mult:\t', multiple_rmse)
  print('Net:\t', neural_rmse)

if __name__ == "__main__":
  main()
