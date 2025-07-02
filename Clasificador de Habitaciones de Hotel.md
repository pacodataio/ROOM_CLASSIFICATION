# Clasificador de Habitaciones de Hotel

## Descripción
Este proyecto implementa un modelo de clasificación para categorizar descripciones de habitaciones de hotel en categorías estandarizadas.

## Archivos Incluidos
- : Script principal para entrenar y evaluar modelos
- : Script para utilizar el modelo entrenado en nuevas descripciones
- : Modelo entrenado serializado
- : Resultados detallados de la evaluación del modelo
- : Ejemplos de predicciones realizadas por el modelo
- : Visualización de la matriz de confusión
- : Visualización de las categorías de habitaciones
- : Datos de muestra utilizados para el entrenamiento
- : Datos etiquetados generados durante el entrenamiento




## Uso del Modelo
### Para clasificar una descripción individual:


### Para clasificar múltiples descripciones desde un archivo CSV:

El archivo CSV debe tener una columna llamada "nombre" con las descripciones de las habitaciones.

## Categorías de Habitaciones
El modelo clasifica las habitaciones en las siguientes categorías:

- APARTMENT: 0
- STANDARD ROOM: 1
- DOUBLE ROOM: 2
- TRIPLE ROOM: 3
- CLASSIC ROOM: 4
- SUPERIOR ROOM: 5
- DELUXE ROOM: 6
- PREMIUM ROOM: 7
- JUNIOR SUITE: 8
- SUITE: 9
- QUADRUPLE ROOM: 10
- STUDIO: 11
- SHARED ROOM: 12
- FAMILY ROOM: 13

## Limitaciones y Recomendaciones
- La precisión actual del modelo está por debajo del objetivo del 98% debido al tamaño limitado de los datos de muestra.
- Para mejorar la precisión, se recomienda:
  1. Aumentar el conjunto de datos de entrenamiento con más ejemplos etiquetados
  2. Refinar las reglas de clasificación para manejar casos ambiguos
  3. Implementar técnicas de aumento de datos para balancear las categorías
  4. Considerar modelos más avanzados de procesamiento de lenguaje natural para el conjunto completo de datos
