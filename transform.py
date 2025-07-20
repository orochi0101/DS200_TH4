import pandas as pd
from sklearn.preprocessing import LabelEncoder

def transform_data(df):
    df_transformed = df.copy()

    le = LabelEncoder()
    df_transformed['Neighborhood'] = le.fit_transform(df_transformed['Neighborhood'])

    df_transformed = df_transformed.fillna(0)

    return df_transformed

