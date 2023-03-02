import pandas as pd

data = pd.read_csv('data/dataset/input.txt')
datal1 = len(data)
data = data.dropna()
datal2 = len(data)
print(datal1, datal2)