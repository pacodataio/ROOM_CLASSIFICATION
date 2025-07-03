import numpy as np
import joblib
import pandas as pd

# Cargar embeddings
X = np.load("modeloTravelGate_embeddings.npy")

# Cargar modelo DBSCAN
dbscan = joblib.load("modeloTravelGate_clusters.pkl")

# Las etiquetas están en dbscan.labels_ (no hay predict)
labels = dbscan.labels_

# Cargar descripciones originales
df = pd.read_csv("habitaciones_sin_etiquetar.csv")
assert 'description' in df.columns, "No se encontró la columna 'description' en el CSV"

# Asignar etiquetas obtenidas por DBSCAN
df['cluster'] = labels
df.to_csv("habitaciones_con_clusters.csv", index=False)
print("✅ DataFrame con clusters guardado en 'habitaciones_con_clusters.csv'")

# Cantidad de clusters (DBSCAN asigna -1 para ruido)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters detectados (excluyendo ruido): {num_clusters}")
print(f"Puntos etiquetados como ruido (cluster = -1): {(labels == -1).sum()}")

# Mostrar ejemplos por cluster (incluyendo ruido)
for cluster_id in sorted(set(labels)):
    count = (labels == cluster_id).sum()
    print(f"\n🔹 Cluster {cluster_id} ({count} elementos):")
    ejemplos = df[df['cluster'] == cluster_id]['description'].head(10).tolist()
    for desc in ejemplos:
        print(f" - {desc}")

# Mostrar todos los ejemplos del cluster -1 (ruido)
'''ruido = df[df['cluster'] == -1]

print(f"\n🔹 Ejemplos del cluster -1 (ruido), total: {len(ruido)}\n")
for desc in ruido['description']:
    print(f" - {desc}")'''
