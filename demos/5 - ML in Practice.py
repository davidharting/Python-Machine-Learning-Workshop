#%% Demo 5 - ML in Practice

#%% Load all libraries
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#%% Set the working directory
os.chdir("C:\\Users\\Matthew\\Dropbox\\Professional\\Workshops\\Practical Machine Learning with Python\\Data")

#%% Read the Iris CSV file
titanic = pd.read_csv("Titanic.csv")

#%% Inspect the data set
titanic.head()

#%% Summarize the columns
titanic.info()

#%% Summarize the data set
titanic.describe(
    include = "all")

#%% Create a correlation matrix
correlations = titanic.corr()

print(correlations)

#%% Create a correlogram
sns.heatmap(
    data = correlations,
    cmap = sns.diverging_palette(
        h_neg = 10, 
        h_pos = 220, 
        as_cmap = True))

#%% Inspect missing values
titanic.isnull().sum()

#%% Assign raw data to temp data set
temp = titanic

#%% Encode sex as one-hot male / female
dummies = pd.get_dummies(temp.sex)
temp = pd.concat([temp, dummies], axis = 1)

#%% Imput missing values for Age
meanAge = temp.age.mean()

temp.age = temp.age.fillna(meanAge)

#%% Engineer new family feature
temp["family"] = temp.sibsp + temp.parch

#%% Encode survived as yes/no
temp.survived.replace((1, 0), ('Yes', 'No'), inplace = True)

#%% Select features
temp = temp.loc[:, ["pclass", "male", "female", "age", "family", "survived"]]

#%% Rename columns
temp.columns = ["Class", "Male", "Female", "Age", "Family", "Survived"]

#%% Assign temp data to clean data
clean = temp

#%% Inspect the cleaned / transformed data
clean.head()

#%% Create features (X)
X = clean.iloc[:, 0:5]

#%% Create labels (y)
y = clean.iloc[:, 5]

#%% Scale the feature data
scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)

#%% Set random seed
np.random.seed(42)

#%% Create stratified training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify = y,
    train_size = 0.80)

#%% Create a knn model
knn_model = KNeighborsClassifier()

#%% Define knn hyper-parameters to test
knn_params = [5, 7, 9, 11, 13]

knn_param_grid = {"n_neighbors" : knn_params }

#%% Create 10 knn models for each of the 5 hyper-parameters
knn_models = GridSearchCV(
    estimator = knn_model, 
    param_grid = knn_param_grid,
    scoring = "accuracy",
    cv = 10,
    verbose = 2)

#%% Train all 50 models
knn_models.fit(
    X = X_train, 
    y = y_train)

#%% Get average accuracy for each hyper-parameter
knn_avg_scores = knn_models.cv_results_["mean_test_score"]

#%% Display average accuracy for each hyper-parameter
for i in range(0, 5):
    print("{:>3} : {:0.3f}"
        .format(knn_params[i], knn_avg_scores[i]))
    
#%% Plot change in accuracy over each hyper-parameter
plt.plot(
    knn_params, 
    knn_avg_scores)
plt.xlabel("k (neighbors)")
plt.ylabel("Accuracy")
plt.show()

#%% Get the top-performing model
knn_top_index = np.argmax(knn_avg_scores)
knn_top_param = knn_params[knn_top_index]
knn_top_score = knn_avg_scores[knn_top_index]
knn_top_error = knn_models.cv_results_["std_test_score"][knn_top_index]

# Inspect the top-performing model
print("Top knn model is k = {:d} at {:0.2f} +/- {:0.3f} accuracy"
    .format(knn_top_param, knn_top_score, knn_top_error))

#%% Create a tree model
tree_model = DecisionTreeClassifier()

#%% Define tree hyper-parameters to test
tree_params = [3, 4, 5, 6, 7]

tree_param_grid = {"max_depth" : tree_params }

#%% Create 10 tree models for each of the 5 hyper-parameters
tree_models = GridSearchCV(
    estimator = tree_model, 
    param_grid = tree_param_grid,
    scoring = "accuracy",
    cv = 10,
    verbose = 2)

#%% Train all 50 models
tree_models.fit(
    X = X_train, 
    y = y_train)

#%% Get average accuracy for each hyper-parameter
tree_avg_scores = tree_models.cv_results_["mean_test_score"]

#%% Display average accuracy for each hyper-parameter
for i in range(0, 5):
    print("{:>3} : {:0.3f}"
        .format(tree_params[i], tree_avg_scores[i]))
    
#%% Plot change in accuracy over each hyper-parameter
plt.plot(
    tree_params, 
    tree_avg_scores)
plt.xlabel("Max Depth (nodes)")
plt.ylabel("Accuracy")
plt.show()

#%% Get the top-performing model
tree_top_index = np.argmax(tree_avg_scores)
tree_top_param = tree_params[tree_top_index]
tree_top_score = tree_avg_scores[tree_top_index]
tree_top_error = tree_models.cv_results_["std_test_score"][tree_top_index]

#%% Inspect the top-performing model's
print("Top tree model is k = {:d} at {:0.2f} +/- {:0.3} accuracy"
    .format(tree_top_param, tree_top_score, tree_top_error))

#%% Create a neual network model
neural_model = MLPClassifier(
    activation = "tanh",
    max_iter = 5000,
    verbose = True)

#%% Define hyper-parameters to test
neural_params = [3, 4, 5, 6, 7]

neural_param_grid = {"hidden_layer_sizes" : neural_params }

#%% Create 10 models for each of the 5 hyper-parameters
neural_models = GridSearchCV(
    estimator = neural_model, 
    param_grid = neural_param_grid,
    scoring = "accuracy",
    cv = 10,
    verbose = 2)

#%% Train all 50 models
neural_models.fit(
    X = X_train, 
    y = y_train)

#%% Get average accuracy for each hyper-parameter
neural_avg_scores = neural_models.cv_results_["mean_test_score"]

#%% Display average accuracy for each hyper-parameter
for i in range(0, 5):
    print("{:>3} : {:0.3f}"
        .format(neural_params[i], neural_avg_scores[i]))
    
#%% Plot change in accuracy over each hyper-parameter
plt.plot(
    neural_params, 
    neural_avg_scores)
plt.xlabel("Hidden Layer Nodes")
plt.ylabel("Accuracy")
plt.show()

#%% Get the top-performing model
neural_top_index = np.argmax(neural_avg_scores)
neural_top_param = neural_params[neural_top_index]
neural_top_score = neural_avg_scores[neural_top_index]
neural_top_error = neural_models.cv_results_["std_test_score"][neural_top_index]

#%% Inspect the top-performing model
print("Top nnet model is k = {:d} at {:0.2f} +/- {:0.3f} accuracy"
    .format(neural_top_param, neural_top_score, neural_top_error))

#%% Compare the top 3 models numerically
print("KNN:  {:0.2f} +/- {:0.3f} accuracy"
    .format(knn_top_score, knn_top_error))
print("Tree: {:0.2f} +/- {:0.3f} accuracy"
    .format(tree_top_score, tree_top_error))
print("NNet: {:0.2f} +/- {:0.3f} accuracy"
    .format(neural_top_score, neural_top_error))

#%% Compare the top three models visually
plt.errorbar(
    x = [knn_top_score, tree_top_score, neural_top_score],
    y = ["KNN", "Tree", "NNet"],
    xerr = [knn_top_error, tree_top_error, neural_top_error],
    linestyle = "none",
    marker = "o")
plt.xlim(0, 1)

#%% Create a final model.
final_model = MLPClassifier(
    hidden_layer_sizes = 7,
    activation = "tanh",
    max_iter = 5000,
    verbose = True)

#%% Train the final model.
final_model.fit(
    X = X_train, 
    y = y_train)

#%% Predict the labels of the hold-out test set.
final_predictions = final_model.predict(X_test)

#%% Get the final prediction accuracy.
final_score = accuracy_score(
    y_true = y_test,
    y_pred = final_predictions)

#%% Inspect the final prediction accuracy.
print(final_score)

#%% Deploy the Model

#%% Question: Will Jack survive the Titanic?

#%% Create an input feature vector for Jack?
X_jack = pd.DataFrame(
    columns = ['Class', 'Male', 'Female', 'Age', 'Family'],
    data = [[3, 1, 0, 20, 0]])

#%% Predict whether Jack survives?
final_model.predict(X_jack)[0]

#%% What is the likelihood that Jack survived?
final_model.predict_proba(X_jack)[0][1]

#%% Just for fun: How likely is it that I survive?
X_matthew = pd.DataFrame(
    columns = ['Class', 'Male', 'Female', 'Age', 'Family'],
    data = [[3, 1, 0, 39, 1]])

final_model.predict_proba(X_matthew)[0][1]