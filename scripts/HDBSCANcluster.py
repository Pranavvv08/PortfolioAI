import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

# Load latest feature snapshot
df = pd.read_csv("data/final_features.csv", index_col=0, parse_dates=True)
latest = df.iloc[-1:]

# Rebuild stock-wise feature matrix
stock_features = {}
for col in latest.columns:
    stock = col.split("_")[0]
    stock_features.setdefault(stock, []).append(col)

stock_matrix = pd.DataFrame([
    latest[cols].values.flatten()
    for stock, cols in stock_features.items()
], index=stock_features.keys())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(stock_matrix)

# UMAP for nonlinear 2D projection
reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
labels = clusterer.fit_predict(X_umap)

# Prepare DataFrame
result = pd.DataFrame(X_umap, columns=["x", "y"])
result["Symbol"] = stock_matrix.index
result["Cluster"] = labels

# Save results
result.to_csv("outputs/HDBSCAN_clustered_stocks.csv", index=False)

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=result, x="x", y="y", hue="Cluster", palette="tab10", s=100)

for i, row in result.iterrows():
    plt.text(row["x"] + 0.1, row["y"] + 0.1, row["Symbol"], fontsize=9)

plt.title("Improved Stock Clustering with UMAP + HDBSCAN")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/HDBSCAN_stock_clusters.png")
plt.show()