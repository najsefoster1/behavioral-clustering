#K-Means Clustering Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

#Load dataset
file_path = "Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv"
if not os.path.exists(file_path):
    print("[ERROR] Dataset not found.")
    exit()

print("[INFO] Loading dataset...")
df = pd.read_csv(file_path)
df = df[df['Data_Value'].notnull()]

#Select and preprocess columns
print("[INFO] Filtering and transforming data...")
columns = ['YearStart', 'LocationAbbr', 'Class', 'Topic', 'Question',
           'StratificationCategory1', 'Stratification1', 'Data_Value']
df = df[columns].dropna()
df_encoded = pd.get_dummies(df.drop('Data_Value', axis=1))

#KMeans - Elbow method to determine optimal clusters
print("[INFO] Running elbow method...")
inertia = []
K_range = range(1, 11)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_encoded)
    inertia.append(model.inertia_)

plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig("elbow_plot.png")
plt.show()

#Run KMeans with chosen number of clusters
print("[INFO] Running KMeans clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_encoded)

#PCA for 2D visualization
print("[INFO] Generating PCA visualization...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_encoded)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('K-Means Clustering (PCA Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.savefig("kmeans_pca_clusters.png")
plt.show()

#statistics per cluster
print("[INFO] Cluster Summary Statistics:")
summary = df.groupby('Cluster')['Data_Value'].describe()
print(summary)

print("\n[INFO] Cluster Descriptions:")
for cluster_id in sorted(df['Cluster'].unique()):
    print(f"\nCluster {cluster_id} insight:")
    temp = df[df['Cluster'] == cluster_id]
    print(f"Avg Health Value: {temp['Data_Value'].mean():.2f}, Entries: {len(temp)}")
    print(f"Most common category: {temp['Class'].mode()[0]}")
