"""
Clasificador de Habitaciones de Hotel - Script de Predicción
==========================================================
Este script permite utilizar el modelo entrenado para clasificar nuevas descripciones
de habitaciones de hotel.
"""

import pickle
import pandas as pd
import sys
import os
import numpy as np
from  model_development import TextPreprocessor

def load_model(model_path):
    """Carga el modelo entrenado"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict_room_type(text, model_data):
    """Predice el tipo de habitación basado en la descripción"""
    best_model = model_data['best_model']
    inv_categories = model_data['inv_categories']

    # Realizar predicción
    prediccion = best_model.predict([text])
    category_id = best_model.predict([text])[0]
    category = inv_categories[int(category_id)]

    confidence = getconfidence(best_model, text)


    return {
        'category_id': int(category_id),
        'category': category,
        'confidence': confidence
    }

def getconfidence(best_model, text):
        """Predice la categoría con nivel de confianza"""


        # Calcular confianza (esto varía según el tipo de modelo)
        confidence = 0.0
        try:
            # Intentar obtener probabilidades de decisión
            if hasattr(best_model, 'predict_proba'):#LogisticRegression, naives Bayes ,RandomForestClassifier,SGDClassifier (log_loss)
                # Para modelos probabilísticos
                proba = best_model.predict_proba([text])
                confidence = np.max(proba)
            elif hasattr(best_model, 'decision_function'): #LogisticRegression,LinearSVC,SGDClassifier (log_loss),PassiveAggressiveClassifier
                # Para SVM y algunos otros
                decision_values = best_model.decision_function([text])
                if len(decision_values.shape) > 1:
                    max_decision = np.max(decision_values)
                    confidence = 1 / (1 + np.exp(-max_decision))  # Sigmoid para normalizar
                else:
                    confidence = 1 / (1 + np.exp(-decision_values[0]))

        except:
            # Si falla, usar un valor por defecto
            confidence = 0.5

        return confidence

def main():
    """Función principal para clasificar habitaciones"""


    # Cargar modelo
    model_path = 'room_classifier_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return

    model_data = load_model(model_path)
    print(f"Modelo cargado: {model_data['best_model_name']}")

    # Procesar entrada
    if model_path =='room_classifier_model1111.pkl':
        if len(sys.argv) < 3:
            print("Error: Debe especificar un archivo CSV")
            return

        # Clasificar desde archivo CSV
        csv_path = sys.argv[2]
        if not os.path.exists(csv_path):
            print(f"Error: No se encontró el archivo {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            if 'nombre' not in df.columns:
                print("Error: El archivo CSV debe tener una columna 'nombre' con las descripciones")
                return

            # Realizar predicciones
            results = []
            for idx, row in df.iterrows():
                prediction = predict_room_type(row['nombre'], model_data)
                results.append({
                    'id': row.get('id', idx),
                    'nombre': row['nombre'],
                    'category_id': prediction['category_id'],
                    'category': prediction['category']
                })

            # Guardar resultados
            results_df = pd.DataFrame(results)
            output_path = 'clasificacion_habitaciones.csv'
            results_df.to_csv(output_path, index=False)
            print(f"Clasificación completada. Resultados guardados en {output_path}")

        except Exception as e:
            print(f"Error al procesar el archivo CSV: {e}")
    else:
        # Clasificar descripción individual
        description = "FULL SIZE BED (1 BIG BED)"
        prediction = predict_room_type(description, model_data)
        print(f"Descripción: '{description}'")
        print(f"Clasificación: {prediction['category']} (ID: {prediction['category_id']})")
        print(f"Confidence: {prediction['confidence']} ")

if __name__ == "__main__":
    main()
