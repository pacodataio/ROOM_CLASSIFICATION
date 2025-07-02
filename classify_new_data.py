# Crear un script classify_new_data.py

from model_development import  RoomClassifier
import pandas as pd


def predict_room_type(text, model_data):
    """Predice el tipo de habitación basado en la descripción"""
    best_model = model_data['best_model']
    inv_categories = model_data['inv_categories']

    # Realizar predicción
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
            if hasattr(best_model, 'decision_function'):
                # Para SVM y algunos otros
                decision_values = best_model.decision_function([text])
                if len(decision_values.shape) > 1:
                    max_decision = np.max(decision_values)
                    confidence = 1 / (1 + np.exp(-max_decision))  # Sigmoid para normalizar
                else:
                    confidence = 1 / (1 + np.exp(-decision_values[0]))
            elif hasattr(best_model, 'predict_proba'):
                # Para modelos probabilísticos
                proba = best_model.predict_proba([text])
                confidence = np.max(proba)
        except:
            # Si falla, usar un valor por defecto
            confidence = 0.5

        return confidence



# Inicializar clasificador
classifier = RoomClassifier()

# Cargar modelo entrenado
classifier.load_model('room_classifier_model.pkl')

# Cargar nuevos datos
df = pd.read_csv('sample_data_big.csv')

# Realizar predicciones con nivel de confianza
results = []
for idx, row in df.iterrows():
    text = row['nombre']
    prediction = classifier.predict(text)
    results.append({
        'id': row.get('id', idx),
        'nombre': text,
        'category': prediction['category'],
        'category_id': prediction['category_id'],
        'confidence': prediction['confidence']
    })

# Guardar resultados
results_df = pd.DataFrame(results)
results_df.to_csv('clasificacion_con_confianza.csv', index=False)

# Identificar casos de baja confianza para revisión
low_confidence = results_df[results_df['confidence'] < 0.7]
low_confidence.to_csv('revisar_clasificaciones.csv', index=False)

print(f"Clasificación completada. Total: {len(results_df)} habitaciones")
print(f"Casos para revisar: {len(low_confidence)} habitaciones")
