# ==========================================================
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("Mall_Customers.csv")

print("=" * 60)
print("First 5 Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

# ----------------------------------------------------------
# Rename 'Genre' to 'Gender' if necessary
# ----------------------------------------------------------

if "Genre" in df.columns:
    df.rename(columns={"Genre": "Gender"}, inplace=True)

# ----------------------------------------------------------
# Remove Duplicate Rows
# ----------------------------------------------------------

df.drop_duplicates(inplace=True)

# ----------------------------------------------------------
# Encode Gender
# Male = 1
# Female = 0
# ----------------------------------------------------------

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

print("\nEncoded Gender Values:")
print(df["Gender"].head())

# ----------------------------------------------------------
# Statistical Summary
# ----------------------------------------------------------

print("\nStatistical Summary")
print(df.describe())

# ----------------------------------------------------------
# Age Distribution
# ----------------------------------------------------------

plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Customers")
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# Income Distribution
# ----------------------------------------------------------

plt.figure(figsize=(6,4))
plt.hist(df["Annual Income (k$)"], bins=20)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Customers")
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# Spending Score Distribution
# ----------------------------------------------------------

plt.figure(figsize=(6,4))
plt.hist(df["Spending Score (1-100)"], bins=20)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Customers")
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# Scatter Plot Before Clustering
# ----------------------------------------------------------

plt.figure(figsize=(7,6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"]
)

plt.title("Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------

X = df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# Elbow Method
# ----------------------------------------------------------

wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.show()

# ----------------------------------------------------------
# KMeans Model
# ----------------------------------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ----------------------------------------------------------
# Silhouette Score
# ----------------------------------------------------------

score = silhouette_score(X_scaled, clusters)

print("\nSilhouette Score:", round(score,3))

# ----------------------------------------------------------
# Cluster Centers
# ----------------------------------------------------------

centers = scaler.inverse_transform(kmeans.cluster_centers_)

centers = pd.DataFrame(
    centers,
    columns=[
        "Age",
        "Annual Income",
        "Spending Score"
    ]
)

print("\nCluster Centers")
print(centers)

# ----------------------------------------------------------
# Cluster Counts
# ----------------------------------------------------------

print("\nCustomers in Each Cluster")
print(df["Cluster"].value_counts())

# ----------------------------------------------------------
# Cluster Summary
# ----------------------------------------------------------

summary = df.groupby("Cluster")[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean()

print("\nCluster Summary")
print(summary)

# ----------------------------------------------------------
# Final Cluster Visualization
# ----------------------------------------------------------

plt.figure(figsize=(8,6))

colors = ["red","blue","green","orange","purple"]

for cluster in sorted(df["Cluster"].unique()):

    temp = df[df["Cluster"] == cluster]

    plt.scatter(
        temp["Annual Income (k$)"],
        temp["Spending Score (1-100)"],
        label=f"Cluster {cluster}"
    )

plt.scatter(
    centers["Annual Income"],
    centers["Spending Score"],
    marker="X",
    s=300,
    label="Centroids"
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)

plt.show()

# ----------------------------------------------------------
# Save Results
# ----------------------------------------------------------

df.to_csv("Customer_Segments.csv", index=False)

print("\nSegmented dataset saved as Customer_Segments.csv")

# ----------------------------------------------------------
# Business Interpretation
# ----------------------------------------------------------

print("\nBUSINESS INSIGHTS")
print("="*60)

for cluster in sorted(df["Cluster"].unique()):

    temp = df[df["Cluster"] == cluster]

    print(f"\nCluster {cluster}")

    print("Customers :", len(temp))

    print("Average Age:",
          round(temp["Age"].mean(),2))

    print("Average Income:",
          round(temp["Annual Income (k$)"].mean(),2))

    print("Average Spending:",
          round(temp["Spending Score (1-100)"].mean(),2))

print("\nProject Completed Successfully!")