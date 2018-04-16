#%% Demo 3 - Regression

#%% Import the OS library
import os

#%% Set the working directory
# os.chdir("C:\\Users\\Matthew\\Dropbox\\Professional\\Workshops\\Practical Machine Learning with Python\\Data")

#%% Import the pandas library
import pandas as pd

#%% Read the Iris CSV file
iris = pd.read_csv("data/Iris.csv")

#%% Import the matplotlib library
import matplotlib.pyplot as plt

#%% Create a scatterplot matrix
pd.plotting.scatter_matrix(
    frame = iris,
    alpha = 1,
    s = 100,
    diagonal = 'none')

#%% Import the seaborn library
import seaborn as sns

#%% Create a correlation matrix
correlations = iris.corr()

print(correlations)

#%% Create a correlogram
sns.heatmap(
    data = correlations,
    cmap = sns.diverging_palette(
        h_neg = 10, 
        h_pos = 220, 
        as_cmap=True))

#%% Get the correlation between petal length and width
iris.Petal_Length \
    .corr(iris.Petal_Width)

#%% Create a scatterplot of petal length vs. width
plt.scatter(
    x = iris.Petal_Length,
    y = iris.Petal_Width)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

#%% Transform the Data

#%% Inspect the data set
iris.head()

#%% Create features (X)
X = iris.iloc[:, 0:3]

#%% Inspect features (X)
X.head()

#%% Convert categorical to one-hot encoding
dummies = pd.get_dummies(X.Species)

#%% Inspect results
dummies.head()

#%% Append the one-hot species feature columns
iris = pd.concat([X, dummies], axis = 1)

#%% Drop the species column_name
X.drop("Species", 1)

#%% Inspect the results
X.head()

#%% Create labels (y)
y = iris.Petal_Width

#%% Inspect labels (y)
y.head()

#%% Create Training and Test Set

#%% Import numpy library
import numpy as np

#%% Set random number seed
np.random.seed(234)

#%% Import train_test_split
from sklearn.model_selection import train_test_split

#%% Split data into 80% training and 20% test 
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    train_size = 0.80)

#%% Inspect training and test sets
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test:  ", X_test.shape)
print("y_test:  ", y_test.shape)

#%% Predict with Simple Linear Regression

#%% Import linear model
from sklearn.linear_model import LinearRegression

#%% Create a linear model
simple_model = LinearRegression()

#%% Extract just the petal length feature (small x)
x1_train = X_train.loc[:, ["Petal_Length"]]
x1_test = X_test.loc[:, ["Petal_Length"]]

#%% Fit the model
simple_model.fit(
    X = x1_train,
    y = y_train)

#%% Draw the regression line on the plot
plt.scatter(
    x = iris.Petal_Length,
    y = iris.Petal_Width,
    color = "black")
plt.plot(
    x1_test,
    simple_model.predict(
        x1_test),
    color = "blue",
    linewidth = 3)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()


#%% Inspect parameter estimates
print("y-intercept (b): ", simple_model.intercept_)
print("Slope (m):        ", simple_model.coef_[0])

#%% Make predictions using the test set
simple_predictions = simple_model.predict(x1_test)

#%% Visualize the prediction error

#%% Plot training set (black dots)
plt.scatter(
    x = x1_train.Petal_Length,
    y = y_train,
    color = "black",
    facecolor = "none")

#%% Plot the predictions (blue dots)
plt.scatter(
    x = x1_test.Petal_Length,
    y = simple_predictions,
    color = "blue",
    marker = 'x')

#%% Plot the correct answer (green dots)
plt.scatter(
    x = x1_test.Petal_Length,
    y = y_test,
    color = "green")

#%% Plot the error (red lines)
plt.plot(
    [x1_test.Petal_Length, x1_test.Petal_Length],
    [simple_predictions, y_test],
    color = "red",
    zorder = 0)

#%% Finish the plot
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

#%% Compute the root mean squared error (RMSE)
simple_rmse = np.sqrt(np.mean((y_test - simple_predictions) ** 2))

print(simple_rmse)

#%% Predict with Multiple Linear Regression

#%% Create a muliple linear regression model
multiple_model = LinearRegression()

#%% Fit the model
multiple_model.fit(
    X = X_train,
    y = y_train)

#%% Inspect parameter estimates
print("{:<12}: {: .3f}"
    .format("y-intercept", multiple_model.intercept_))

for i, column_name in enumerate(X_train.columns):
    print("{:<12}: {: .3f}".format(
        column_name, 
        multiple_model.coef_[i]))

#%% Make predictions using the test set
multiple_predictions = multiple_model.predict(X_test)

#%% Visualize the prediction error
plt.scatter(
    x = X_train.,
    y = y_train,
    color = "black",
    facecolor = "none")
plt.scatter(
    x = x_test,
    y = multiple_predictions,
    color = "blue",
    marker = 'x')
plt.scatter(
    x = x_test,
    y = y_test,
    color = "green")
plt.plot(
    [x_test.reshape(-1), x_test.reshape(-1)],
    [multiple_predictions, y_test],
    color = "red",
    zorder = 0)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

#%% Compute the root mean squared error (RMSE)
multiple_rmse = np.sqrt(np.mean((y_test - multiple_predictions) ** 2))

print(multiple_rmse)

#%% Predict with a Neural Network Regressor

#%% Import standard scaler
from sklearn.preprocessing import StandardScaler

#%% Create a standard scaler
X_scaler = StandardScaler()
y_scaler = StandardScaler()

#%% Fit the scaler to all training data
X_scaler.fit(X)
y_scaler.fit(y.values.reshape(-1, 1))

#%% Scale the training and test data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

#%% Import the neural network regressor library
from sklearn.neural_network import MLPRegressor

#%% Create a neural network regressor
neural_model = MLPRegressor(
    hidden_layer_sizes = (4),
    activation = "tanh",
    solver = "lbfgs",
    max_iter = 1000,
    verbose = True)

#%% Train the model
neural_model.fit(
    X = X_train_scaled,
    y = y_train_scaled.reshape(-1, ))

#%% Predict with the model
scaled_predictions = neural_model.predict(X_test_scaled)

#%% Unscale the predictions
neural_predictions = y_scaler.inverse_transform(scaled_predictions)

#%% Visualize the prediction error
plt.scatter(
    x = X_train.Petal_Length,
    y = y_train,
    color = "black",
    facecolor = "none")
plt.scatter(
    x = X_test.Petal_Length,
    y = neural_predictions,
    color = "blue",
    marker = 'x')
plt.scatter(
    x = X_test.Petal_Length,
    y = y_test,
    color = "green")
plt.plot(
    [X_test.Petal_Length, X_test.Petal_Length],
    [neural_predictions, y_test],
    color = "red",
    zorder = 0)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

#%% Compute the root mean squared error (RMSE)
neural_rmse = np.sqrt(np.mean((y_test - neural_predictions) ** 2))

print(neural_rmse)

#%% Compare all three results
print("Simple RMSE:   ", simple_rmse)
print("Multiple RMSE: ", multiple_rmse)
print("Neural RMSE:   ", neural_rmse)
