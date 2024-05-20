import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load embeddings and compute cosine similarity matrix
def compute_cosine_similarity(file_path):
    # Load embeddings
    df = pd.read_csv(file_path)
    # Assuming the first column 'label' needs to be dropped, adjust if necessary
    if 'label' in df.columns:
        features = df.drop('label', axis=1).values
    else:
        features = df.values
    
    # Compute cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(features)
    return cosine_sim_matrix

# File paths
bert_path = "D:\\College\\Sem_4\\Machine Learning\\Project\\Final Project Files\\poems_embeddings_bert.csv"
gpt2_path = "D:\\College\\Sem_4\\Machine Learning\\Project\\Final Project Files\\poems_embeddings_GPT2.csv"
roberta_path = "D:\\College\\Sem_4\\Machine Learning\\Project\\Final Project Files\\poems_embeddings_roberta.csv"

# Compute cosine similarity matrices
bert_cosine_sim = compute_cosine_similarity(bert_path)
gpt2_cosine_sim = compute_cosine_similarity(gpt2_path)
roberta_cosine_sim = compute_cosine_similarity(roberta_path)

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(bert_cosine_sim, ax=ax[0], cmap='viridis')
ax[0].set_title('BERT Cosine Similarity Matrix')
ax[0].set_xlabel('Poems')
ax[0].set_ylabel('Poems')

sns.heatmap(gpt2_cosine_sim, ax=ax[1], cmap='viridis')
ax[1].set_title('GPT-2 Cosine Similarity Matrix')
ax[1].set_xlabel('Poems')
ax[1].set_ylabel('Poems')

sns.heatmap(roberta_cosine_sim, ax=ax[2], cmap='viridis')
ax[2].set_title('RoBERTa Cosine Similarity Matrix')
ax[2].set_xlabel('Poems')
ax[2].set_ylabel('Poems')

plt.tight_layout()
plt.show()