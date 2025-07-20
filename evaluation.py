import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import pickle
import transform  

test_path = "/home/mthang/ds200th/th2/processed/test.csv"
model_path = "/home/mthang/ds200th/th2/model_randomforestregressor.pkl"
output_path = "/home/mthang/ds200th/th2/evaluation_results.txt"

df_test = pd.read_csv(test_path)

df_test_transformed = transform.transform_data(df_test)

X_test = df_test_transformed.drop(columns=["Price"])
y_test = df_test_transformed["Price"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

with open(output_path, "w") as f:
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R² Score: {r2:.2f}\n")

print(f"Saved results!")


