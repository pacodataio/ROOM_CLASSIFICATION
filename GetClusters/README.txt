MODEL: DISTILBERT from scratch (distilbert-base-multilingual-case)


1. El objetivo es intenta crear nuestra propias etiquetas/categorias a partir de datos sin etiquetar,
 para eso se hace un clustering con kmeans para intentar detectar patrones. El modelo agrupa los datos en cluster  de acuerdo patrones que detecta en los datos.
 Una vez definidos los cluster, se observan los datos por cluster, y se renombra los cluster de acuerdo a las categorias que queremos:
 cluster 0 -> "APARTMENT"
 cluster 1->  "STANDARD ROOM",
 cluster 2: "DOUBLE ROOM",
 cluster 3: "TRIPLE ROOM", etc
Esto depende de nuestra observacion, y es  un trabajo mas manual, pero mas simplp si tuviera que etiquetarse los  3000  o mas datos de entrenamiento.
El  desafio es  conseguir el clustering idoneo.

PROCESO:

1. Usa clustering para agrupar textos similares
Técnicas como KMeans o DBSCAN sobre vectores de texto (embeddings) pueden ayudarte a encontrar grupos naturales.

Por ejemplo, puedes usar un modelo preentrenado para generar embeddings de tus textos y luego hacer clustering.

Así detectas grupos de habitaciones similares sin etiquetas previas.
*** GET_ROOM_Cluster_kmeans:
 Ejecutar un análisis de Elbow o Silhouette para elegir mejor el número.
    from sklearn.metrics import silhouette_score
    for k in range(5, 20):
        kmeans_tmp = KMeans(n_clusters=k, random_state=42)
        labels_tmp = kmeans_tmp.fit_predict(X)
        score = silhouette_score(X, labels_tmp)
        print(f"k={k}, silhouette_score={score:.3f}")

*** GET_ROOM_Cluster_dbscan: opcion 2 , clsutering usando dbscan
2. Explorar y valida los clusters manualmente
Mira muestras de cada cluster.

Asigna etiquetas manualmente a esos grupos (por ejemplo: cluster 0 = "standard", cluster 1 = "suite", etc.).

Esto te da un set de datos etiquetados semi-automáticamente.

3. Usar esos datos para hacer fine-tuning supervisado
Con esas etiquetas ahora sí entrenas un modelo de clasificación.
Esto mejora precisión y permite generalizar mejor.

4. Después de asignar etiquetas a clusters, puedes crear CSV para fine-tuning

RESUMEN:

| Paso                        | Qué haces                                  |
| --------------------------- | ------------------------------------------ |
| **Generar embeddings**      | Convierte textos a vectores numéricos      |
| **Clustering**              | Agrupa textos similares sin etiquetas      |
| **Revisión manual**         | Etiqueta clusters para tener clases claras |
| **Fine-tuning supervisado** | Entrena modelo con esas etiquetas          |
