import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
import pickle
import transform
train_path = "/home/mthang/ds200th/th2/processed/train.csv"

df = pd.read_csv(train_path)

df_transformed = transform.transform_data(df)

X = df_transformed.drop(columns=["Price"])
y = df_transformed["Price"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

model_path = "/home/mthang/ds200th/th2/model_randomforestregressor.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Saved model!")


