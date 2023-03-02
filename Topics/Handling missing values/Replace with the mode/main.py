import pandas as pd

data = pd.read_csv("data/dataset/input.txt")
# Find the categorical feature with missing values
cat_feature = "location"
# Fill NaNs with the mode
data[cat_feature].fillna(data[cat_feature].mode()[0], inplace=True)
print(data.head(5))