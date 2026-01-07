import pandas as pd

# Load dataset
df = pd.read_csv("products.csv")

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Create a combined feature for recommendation
df['combined_features'] = (
    df['category'] + " " +
    df['description']
)

# Preview cleaned data
print(df[['product_name', 'combined_features']].head())

# Save cleaned data
df.to_csv("cleaned_products.csv", index=False)

print("\nCleaned dataset saved successfully!")
