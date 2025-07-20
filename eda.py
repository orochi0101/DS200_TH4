import pandas as pd
import os

input_path = "/home/mthang/ds200th/th2/housing_price_dataset.csv" 

df = pd.read_csv(input_path)

print("First 5 lines:")
print(df.head().to_string())

print("Check for null value:")
print(df.isnull().sum().to_string())

print("Column data type:")
print(df.dtypes.to_string())


