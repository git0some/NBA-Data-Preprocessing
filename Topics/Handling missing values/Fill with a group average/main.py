#  write your code here 
import pandas as pd

# load the data
data = pd.read_csv("data/dataset/input.txt")

# calculate the average height by location
avg_height_by_location = data.groupby("location")["height"].mean()

# fill NaNs with average height by location
data["height"].fillna(data["location"].map(avg_height_by_location), inplace=True)
data["height"] = data["height"].round(1)

# calculate the sum of the height column
height_sum = data["height"].sum()

# print the result
print(height_sum)

