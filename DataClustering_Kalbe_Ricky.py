import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# scalling
from sklearn.preprocessing import StandardScaler
# modelling
from sklearn.cluster import KMeans
#silhoute
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

# Or, only disable specific warnings based on category
# Example: Disabling DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
df = pd.read_csv('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist/merged_data.csv')
df.head()
df.info()
df.shape
aggregated_data = df.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()
aggregated_data.rename(columns={'TransactionID' : "TotalTransaction", 'Qty' : 'TotalQuantity'}, inplace=True)
aggregated_data.head()
aggregated_data.info()
aggregated_data.describe()
sns.pairplot(data = aggregated_data)
# Menghitung korelasi antara variabel
correlation_matrix = aggregated_data[['TotalTransaction', 'TotalQuantity', 'TotalAmount']].corr()

# Membuat heatmap untuk visualisasi korelasi dengan colormap 'RdBu_r'
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0)
plt.title('Correlation Heatmap')
plt.show()
# Scaling
sc = StandardScaler()
dfoutlier_std = sc.fit_transform(aggregated_data[['TotalTransaction','TotalQuantity','TotalAmount']].astype(float))

new_dfoutlier_std = pd.DataFrame(data = dfoutlier_std, columns = ['TotalTransaction','TotalQuantity','TotalAmount'])
new_dfoutlier_std.head()
sns.pairplot(new_dfoutlier_std)
# Elbow Method
# declare Within-Cluster Sum of Squares (wcss)
wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init='k-means++', random_state = 42)
  kmeans.fit(new_dfoutlier_std)
  wcss.append(kmeans.inertia_)
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
# Silhoute Method
for i in range(2,13):
    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=42).fit(new_dfoutlier_std).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(new_dfoutlier_std,labels,metric="euclidean",random_state=42)))
# k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(new_dfoutlier_std)
labels = kmeans.labels_
new_dfoutlier_std['label_kmeans'] = labels
new_dfoutlier_std.head()
# Define colors and markers for clusters
colors_cluster = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

# PLOTTING
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for points in clusters
for cluster_id in range(3):
    cluster_data = new_dfoutlier_std[new_dfoutlier_std["label_kmeans"] == cluster_id]
    ax.scatter(cluster_data["TotalTransaction"], cluster_data["TotalQuantity"], cluster_data["TotalAmount"],
               c=colors_cluster[cluster_id], s=50, marker=markers[cluster_id], edgecolor='black',
               alpha=0.7, label=label_cluster[cluster_id])

# Scatter plot for cluster centers (red)
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=250, marker='X', edgecolor='black', label='Cluster Centers')

# Set labels and legend
ax.set_xlabel('TotalTransaction')
ax.set_ylabel('TotalQuantity')
ax.set_zlabel('TotalAmount')
ax.legend()

# Customize grid lines and background color
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_facecolor('#f0f0f0')

plt.show()

# copy label cluster to origin data
df_customer_clustering = aggregated_data.copy()
df_customer_clustering['cluster'] = kmeans.labels_
df_customer_clustering.head()
# Calculate the average metrics for each cluster, including customer count, mean TotalTransaction, mean TotalQuantity, and mean TotalAmount.
df_customer_clustering.groupby('cluster').agg({'CustomerID':'count',
                                               'TotalTransaction':'mean',
                                               'TotalQuantity':'mean',
                                               'TotalAmount':'mean'
                                               }).sort_values(by='TotalAmount').reset_index()
# save csv
df_customer_clustering.to_csv('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist/df_customer_clustering.csv',index=False)
# Set style and define a color palette
sns.set(style="whitegrid")
palette = sns.color_palette("husl", 3)  # You can choose a different palette if desired

# Create scatter plots for each feature pair
feature_pairs = [('TotalTransaction', 'TotalQuantity'), ('TotalTransaction', 'TotalAmount'), ('TotalQuantity', 'TotalAmount')]

for pair in feature_pairs:
    plt.figure(figsize=(10, 6))
    for cluster_num in range(len(palette)):
        cluster_data = new_dfoutlier_std[new_dfoutlier_std.label_kmeans == cluster_num]
        plt.scatter(cluster_data[pair[0]], cluster_data[pair[1]], color=palette[cluster_num], label=f'Cluster {cluster_num}', alpha=0.7, edgecolors='k')
    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.title(f'Scatter Plot of {pair[0]} vs {pair[1]}')
    plt.legend()
    plt.show()
