import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load log returns (price-based correlation)
log_returns = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)

# Compute Pearson correlation matrix
corr_matrix = log_returns.corr(method='pearson')

# Save correlation matrix
os.makedirs("outputs", exist_ok=True)
corr_matrix.to_csv("outputs/correlation_matrix.csv")
print("Saved correlation matrix to outputs/correlation_matrix.csv")

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Stock Correlation Matrix (Pearson)")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()
