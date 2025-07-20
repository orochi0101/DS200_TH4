import pandas as pd
import os

input_path = "/home/mthang/ds200th/th2/housing_price_dataset.csv"
output_dir = "/home/mthang/ds200th/th2/processed/"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

train_df = df.sample(frac=0.7, random_state=42)
remaining_df = df.drop(train_df.index)
test_df = remaining_df.sample(frac=0.6667, random_state=42)  # 20% of total
demo_df = remaining_df.drop(test_df.index)

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
demo_df.to_csv(os.path.join(output_dir, "demo.csv"), index=False)

print("Saved successfully!")


