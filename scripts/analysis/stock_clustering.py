import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load final features (latest date snapshot)
df = pd.read_csv("data/final_features.csv", index_col=0, parse_dates=True)
latest_snapshot = df.iloc[-1:]

# Group feature columns by stock
stock_features = {}
for col in latest_snapshot.columns:
    stock = col.split("_")[0]
    if stock not in stock_features:
        stock_features[stock] = []
    stock_features[stock].append(col)

# Build stock-wise feature matrix
stock_matrix = pd.DataFrame([
    latest_snapshot[cols].values.flatten()
    for stock, cols in stock_features.items()
], index=stock_features.keys())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(stock_matrix)

# Try clustering with KMeans
k = 5  # you can tune this
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Apply PCA for plotting
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(X_scaled)

# Prepare DataFrame
cluster_df = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])
cluster_df["Symbol"] = stock_matrix.index
cluster_df["Cluster"] = cluster_labels

# Save to CSV
cluster_df.to_csv("outputs/clustered_stocks.csv", index=False)

# Plot clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=100)

for i, row in cluster_df.iterrows():
    plt.text(row["PC1"] + 0.02, row["PC2"] + 0.02, row["Symbol"], fontsize=9)

plt.title("Clustering of NIFTY 50 Stocks")
plt.tight_layout()
plt.grid(True)
plt.savefig("outputs/stock_clusters.png")
plt.show()
