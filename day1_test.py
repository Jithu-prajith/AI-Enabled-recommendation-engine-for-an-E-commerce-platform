import pandas as pd

df = pd.read_csv("products.csv")

print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns)