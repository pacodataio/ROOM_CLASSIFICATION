import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN #en el caso de que el clsuering se haga con dbscan

# --------------------------
# 1. Cargar y limpiar datos
# --------------------------
df = pd.read_csv("habitaciones_sin_etiquetar.csv")  # Cambiar por el archivo real
assert 'description' in df.columns, "La columna 'description' es obligatoria."

def limpiar_texto(texto):
    return str(texto).strip().lower()

df['cleaned_description'] = df['description'].apply(limpiar_texto)

# ------------------------------------------------------
# 2. Obtener embeddings con DistilBERT multilingüe
# ------------------------------------------------------
print("Cargando tokenizer y modelo...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
model.eval()

def obtener_embedding(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

print("Generando embeddings...")
embeddings = [obtener_embedding(desc) for desc in tqdm(df['cleaned_description'])]
X = np.array(embeddings)


# ----------------------------------------
# 3. Buscar número óptimo de clusters con Silhouette Score
# ----------------------------------------
# --- DBSCAN parameters ---
eps = 1.2        # Distancia máxima para que 2 puntos sean vecinos
min_samples = 5  # Mínimo de puntos para formar un cluster

# --- Aplicar DBSCAN ---
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
labels = dbscan.fit_predict(X)
# --- Agregar resultados al DataFrame ---
df['cluster_dbscan'] = labels  # -1 significa "ruido" (no asignado a ningún cluster)

# --- Analizar resultados ---
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"🧠 DBSCAN detectó {n_clusters} clusters.")
print(f"❌ DBSCAN detectó {n_noise} puntos de ruido.")

# Silhouette Score (solo si más de 1 cluster)
if n_clusters > 1:
    score = silhouette_score(X[labels != -1], labels[labels != -1])
    print(f"📊 Silhouette Score (sin ruido): {score:.4f}")
else:
    print("⚠️ No hay suficientes clusters para calcular Silhouette Score.")

# --- Opcional: ver algunos ejemplos ---
print(df[['description', 'cluster_dbscan']].groupby('cluster_dbscan').head(3))



# ----------------------------------------
# 5. Entrenar modelo supervisado
#    Clasificación supervisada con RandomForest
# ----------------------------------------
print("Entrenando clasificador supervisado...")
X_train, X_test, y_train, y_test = train_test_split(X, df['cluster_dbscan'], test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ----------------------------------------
# 6. Evaluación del modelo
# ----------------------------------------
y_pred = clf.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print (y_pred)

# ----------------------------------------
# 6. Guardado del modelo
# ----------------------------------------
joblib.dump(clf, "modeloTravelGate_clasificador.pkl")
joblib.dump(dbscan, "modeloTravelGate_clusters.pkl")
np.save("modeloTravelGate_embeddings.npy", X)

print("✅ Modelo y artefactos guardados correctamente.")
