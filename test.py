import pandas as pd

# Load
df = pd.read_csv("data/training_dataset.csv", index_col=0)

# Show shape
print("📊 Dataset Shape:", df.shape)

# Show columns (features, cluster, target)
print("\n🧩 Feature Columns Sample:")
print(df.columns[:10])

# Show a few rows
print("\n🔍 Sample Rows:")
print(df.head())

# Check for NaNs
print("\n🧼 Any NaNs?")
print(df.isna().sum().sum())
