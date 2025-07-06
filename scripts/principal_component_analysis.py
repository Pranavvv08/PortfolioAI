import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load engineered features
df = pd.read_csv("data/final_features.csv", index_col=0, parse_dates=True)

# Step 1: Take last available day’s features (i.e., most recent market snapshot)
latest_snapshot = df.iloc[-1:]  # shape = (1, num_features)

# Step 2: Transpose to stock-wise format
# Columns are like 'RELIANCE_mom_5d', 'TCS_vol_20d' etc.
# We’ll group features by stock
stock_features = {}

for col in latest_snapshot.columns:
    stock = col.split("_")[0]
    if stock not in stock_features:
        stock_features[stock] = []
    stock_features[stock].append(col)

# Create a DataFrame where each row is a stock, columns are its engineered features
stock_matrix = pd.DataFrame([
    latest_snapshot[cols].values.flatten()
    for stock, cols in stock_features.items()
], index=stock_features.keys())

# Step 3: Standardize the data
scaler = StandardScaler()
stock_scaled = scaler.fit_transform(stock_matrix)

# Step 4: Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(stock_scaled)

# Step 5: Visualize
pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
pca_df["Symbol"] = stock_matrix.index

plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=100)

for i, row in pca_df.iterrows():
    plt.text(row["PC1"] + 0.02, row["PC2"] + 0.02, row["Symbol"], fontsize=9)

plt.title("📊 PCA Visualization of NIFTY 50 Stock Behaviors")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/pca_visualization.png")
plt.show()
