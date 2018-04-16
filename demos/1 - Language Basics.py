# Demo 1 - Language Basics

#%% Assign a variable
x = "Hello World"

#%% Implicitly print a variable
print(x)
x

#%% Create a boolean variable
b = True

print(b)

#%% Create a integer variable
i = 123

print(i)

#%% Create a numeric (float-point) variable
f = 2.34

print(f)

#%% Create a character string variable
c = "ABC 123"

print(c)

#%% Define a function
def add(a, b):
    return a + b

#%% Invoke a function
add(1, 2)

#%% Using Built-in Data Structures

#%% Create a tuple
t = (1, 2, 3)

#%% Print a tuple
print(t)

#%% Access an element of a tuple
t[0]

#%% Create a list
l = [1, 2, 3]

#%% Print a list
print(l)

#%% Access an element of a list
l[1]

#%% Create a dictionary
d = {"a" : 1, "b" : 2, "c" : 3}

#%% Print a dictionary
print(d)

#%% Access an entry in a dictionary
d["c"]

# NOTE: Tuples are immutable, lists are mutable
# NOTE: both are hetrogenius

#%% Using Numpy Data Structures

#%% Import numpy package
import numpy as np

#%% Create an array
a = np.array([1, 2, 3])

#%% Print an array
print(a)

#%% Access an element of an array
a[1]

#%% Create a sequence
s = np.arange(1, 6)

#%% Print a sequence
print(s)

# Note end of range is exclusive

#%% Create a 2x3 matrix
m = np.matrix([[1, 2, 3], [4, 5, 6]])

#%% Print a matrix
print(m)

#%% Access an element in a matrix
m[0, 1]

#%% Create a 2x2x2 array
cube = np.array([[[1, 2], [2, 3]], [[4, 5], [6, 7]]])

#%% Print a 3D array
print(cube)

#%% Access an element of a 3D array
cube[0, 0, 1]

#%% Perform a vectorized operation
np.array([1, 2, 3]) + np.array([2, 4, 6])

# Using Pandas Data Structures

#%% Import pandas
import pandas as pd

#%% Create a dataframe (row-wise)
df = pd.DataFrame(
    columns = ["Name", "How_Many", "Is_Pet"],
    data = [["Cat", 5, True],
            ["Dog", 10, True],
            ["Cow", 15, False],
            ["Pig", 20, False]])

#%% NOTE: Alternate version (column-wise)
df = pd.DataFrame(
    columns = ["Name", "How_Many", "Is_Pet"],
    data = {"Name" : ["Cat", "Dog", "Cow", "Pig"],
            "How_Many" : [5, 10, 15, 20],
            "Is_Pet" : [True, True, False, False]})
    
#%% Print a dataframe
print(df)

#%% Index a dataframe by row and column
df.iloc[0, 1]

#%% Index by row
df.iloc[0, :]

#%% Index by column
df.iloc[:, 1]

#%% Index by column name
df.loc[:, "How_Many"]

#%% Subset by list of row indexes
df.iloc[[1, 3], :]

#%% Subset by sequence of row indexes
df.iloc[1:4, :]

#%% Subset by list of booleans
df.iloc[[True, False, True, False], :]

#%% Execute a predicate function
df.IsPet == True

#%% Subset using a predict function
df[df.IsPet == True]

#%% Subset using inequality
df[df.HowMany > 12]

#%% Subset using more complex function
df[df.Name.isin(["Cat", "Cow"])]

#%% Using Named vs ordered arguments
m1 = np.matrix(
    data = [[1, 2], [3, 4]],
    dtype = "float")

m2 = np.matrix([[1, 2], [3, 4]], "float")

m1 == m2

#%% Find help on a function
help(np.matrix)

# TODO: Installing modules with pip / conda