Fecha: 2025-06-13
Estado del proyecto:

- Datos originales: habitaciones_sin_etiquetar.csv
- Embeddings guardados: modeloTravelGate_embeddings.npy
- Modelo clustering guardado: modeloTravelGate_clusters.pkl (DBSCAN, eps=1.2, min_samples=5)
- Modelo clasificador guardado: modeloTravelGate_clasificador.pkl (RandomForest)
- Resultados: muchos puntos de ruido (-1) en DBSCAN (~95%)
- DataFrame con clusters guardado en habitaciones_con_clusters.csv

Próximos pasos:
- Evaluar ajuste de parámetros DBSCAN o probar otro algoritmo
- Revisar manualmente clusters para decidir etiquetado final
- Entrenar modelo supervisado con clusters finales
