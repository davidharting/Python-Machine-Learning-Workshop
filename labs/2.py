import pandas as pd

def main():
  policies = pd.read_csv("data/Risk.csv")
  print(policies)

if __name__ == "__main__":
  main()
