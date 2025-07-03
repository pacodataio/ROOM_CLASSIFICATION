import numpy as np
import joblib
import pandas as pd

# Cargar embeddings
X = np.load("modeloTravelGate_embeddings.npy")

# Cargar modelo KMeans
kmeans = joblib.load("modeloTravelGate_clusters.pkl")


# Cargar las descripciones originales
df = pd.read_csv("habitaciones_sin_etiquetar.csv")  # Asegúrate de que esta columna exista: 'description'

# Asignar etiquetas del modelo
df['cluster'] = kmeans.predict(X)

# Mostrar ejemplos por cluster
num_clusters = len(np.unique(df['cluster']))
print(f"clusters:{num_clusters}")
for i in range(num_clusters):
    print(f"\n🔹 Cluster {i}:")
    ejemplos = df[df['cluster'] == i]['description'].head(20).to_list()
    for desc in ejemplos:
        print(f" - {desc}")

