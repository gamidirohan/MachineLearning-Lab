import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the embeddings from the CSV file
df = pd.read_csv("D:\College\Sem_4\Machine Learning\Project\Final Project Files\poems_embeddings_bert.csv")
features = df.drop('label', axis=1).values  # Assuming 'label' is the column with actual class labels
actual_labels = df['label'].values

# Since the data is already in the correct shape (n_samples, n_features), no reshaping is necessary

# Initialize and fit KMeans algorithm
k = 8  # As specified to use 8 clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(features)

# Get cluster assignments from KMeans
predicted_labels = kmeans.labels_

# Compute the confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

# Create a DataFrame from the confusion matrix for visualization
cm_df = pd.DataFrame(cm, index=[f'Class {i+1}' for i in range(k)],
                     columns=[f'Cluster {i+1}' for i in range(k)])

# Display the DataFrame
print("Confusion Matrix: Actual Classes vs KMeans Clusters")
print(cm_df)

# Optionally plot the confusion matrix for better visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Actual Classes vs KMeans Clusters")
plt.ylabel("Actual Classes")
plt.xlabel("KMeans Clusters")
plt.show()