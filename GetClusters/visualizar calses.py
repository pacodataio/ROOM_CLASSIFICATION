import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Cargar embeddings (numpy array)
X = np.load("modeloTravelGate_embeddings.npy")

# Cargar modelo KMeans para obtener las etiquetas de cluster
kmeans = joblib.load("modeloTravelGate_clusters.pkl")

# Obtener etiquetas de cluster para los embeddings
clusters = kmeans.predict(X)

sns.set(style="whitegrid")

# --- PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='tab10', s=50)
plt.title("Clusters visualizados con PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend(title="Cluster")

# --- t-SNE ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=clusters, palette='tab10', s=50)
plt.title("Clusters visualizados con t-SNE")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.legend(title="Cluster")

plt.tight_layout()
plt.show()