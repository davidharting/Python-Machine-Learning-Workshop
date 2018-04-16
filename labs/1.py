# Simple introduction to python

import numpy as np
import pandas as pd

def variables():
  x = "Hello world"   # String
  print(x)

  b = True # Boolean
  print(b)

  i = 123 # Integer
  print(i)

  f = 2.34  # Float
  print(f)

  c = "ABC 123"
  print(c)

def add(a, b):
  return a +b

def dataStructures():
  t = (1, 2, 3) # tuple
  print(t)
  print(t[0])

  l = [2, 3, 4] # list
  print(l)
  print(l[2])

  d = { "a":1, "b":2, "c":3 } # dictionary
  print(d)
  print(d['b'])

def numpyDataStructures():
  print('\n\nNumpy Data Structures')
  a = np.array([1, 2, 3])
  print(a)
  print(a[0])

  range = np.arange(1, 6)
  print(range)

  m = np.matrix([
    [1, 2, 3],
    [4, 5, 6]
  ])
  print(m)
  print(m[0, 1])

def pandasDataStructures():
  print('\n\nPandas Data Structures')
  df = pd.DataFrame(
    columns = ["Name", "How_Many", "Is_Pet"],
    data=[
      ["Cat", 5, True],
      ["Dog", 10, True],
      ["Cow", 15, False],
      ["Pig", 20, False]
    ]
  )
  print(df)
  print(df.iloc[0, 1]) # Middle column of Cat row
  print('\n', df.iloc[0, :]) # Entire cat row
  print('\n', df.iloc[:, 0]) # Name column for all rows
  print('\n', df.loc[:, "How_Many"]) # Entire How_Many column for all rows, by name

  # Subsetting
  print(df.iloc[[1, 3], :]) # Grab all columns of rows 1 *and* 3
  print('\n', [df.iloc[1:4, :]]) # Grab rows 1-3 (sequence)
  print(df.iloc[[True, False, True, False], :]) # Individual test each row for inclusion
  print(df[df.Is_Pet == True]) # General test for inclusion
  print(df[df.How_Many > 12])
  print(df[df.Name.isin(["Cow", "Cat"])])

def matrices():
  print('\n\nMatrices')
  # Named parameters
  m1 = np.matrix(
    data = [
      [1, 2],
      [3, 4]
    ],
    dtype = "float"
  )
  # ordered parameters
  m2 = np.matrix(
    [[1, 2], [3, 4]],
    "float"  
  )
  # All matrix operations yield a matrix
  # including equality
  print(m1 == m2)

def main():
  variables()
  print(add(1, 2))
  dataStructures()
  numpyDataStructures()
  pandasDataStructures()
  matrices()

if __name__ == "__main__":
  main()
