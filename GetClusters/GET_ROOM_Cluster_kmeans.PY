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
sil_scores = []
k_values = range(10, 30)  # Prueba desde 5 hasta 20 clusters

print("Evaluando distintos números de clusters para KMeans...")
for k in k_values:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42)
    labels_tmp = kmeans_tmp.fit_predict(X)
    score = silhouette_score(X, labels_tmp)
    sil_scores.append(score)
    print(f"k={k}, silhouette_score={score:.4f}")

# Graficar resultados
plt.figure(figsize=(8,4))
plt.plot(k_values, sil_scores, marker='o')
plt.xlabel('Número de clusters k')
plt.ylabel('Silhouette Score')
plt.title('Elección del número óptimo de clusters')
plt.show()

# Elegir k con mayor Silhouette Score
best_k = k_values[np.argmax(sil_scores)]
print(f"\nNúmero óptimo de clusters según Silhouette Score: {best_k}")

# ----------------------------------------
# 4. Clustering final con el mejor k
# ----------------------------------------
print(f"Entrenando KMeans definitivo con k={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
print(df[['description', 'cluster']].head())

# Silhouette Score final
final_score = silhouette_score(X, df['cluster'])
print(f"Silhouette Score final: {final_score:.4f}")

# ----------------------------------------
# 5. Entrenar modelo supervisado
#    Clasificación supervisada con RandomForest
# ----------------------------------------
print("Entrenando clasificador supervisado...")
X_train, X_test, y_train, y_test = train_test_split(X, df['cluster'], test_size=0.2, random_state=42)
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
joblib.dump(kmeans, "modeloTravelGate_clusters.pkl")
np.save("modeloTravelGate_embeddings.npy", X)

print("✅ Modelo y artefactos guardados correctamente.")
