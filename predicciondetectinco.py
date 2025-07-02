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
    category_id = best_model.predict([text])[0]
    category = inv_categories[int(category_id)]

    return {
        'category_id': int(category_id),
        'category': category
    }

def main():
    """Función principal para clasificar habitaciones con deteccion de mal clasificados"""
    # se neceita que el archivo venga con  datos ya etiquetados
    #debe incluirse la columna categoria_real

    print (len(sys.argv),":", sys.argv[0] )
    """Función principal para clasificar habitaciones"""
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("Uso: python predict_room.py <descripción> o python predict_room.py --file <archivo_csv>")
        return

    # Cargar modelo
    model_path = 'room_classifier_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return

    model_data = load_model(model_path)
    print(f"Modelo cargado: {model_data['best_model_name']}")

    # Procesar entrada
    if sys.argv[1] == '--file':
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
            correct_count = 0
            incorrect_examples = []

            for idx, row in df.iterrows():
                prediction = predict_room_type(row['nombre'], model_data)
                result=({
                    'id': row.get('id', idx),
                    'nombre': row['nombre'],
                    'category_id': prediction['category_id'],
                    'category': prediction['category']
                })

                # Si el CSV tiene una columna 'category_real', comparar con la predicción
                if 'category_real' in df.columns:
                    result['category_real'] = row['category_real']
                    if str(row['category_real']).upper() == prediction['category'].upper():
                        result['correct'] = True
                        correct_count += 1
                    else:
                        result['correct'] = False
                        incorrect_examples.append(result)

                results.append(result)

            # Guardar resultados
            results_df = pd.DataFrame(results)
            output_path = 'clasificacion_habitaciones.csv'
            results_df.to_csv(output_path, index=False)
            print(f"Clasificación completada. Resultados guardados en {output_path}")

            # Mostrar estadísticas si hay categorías reales
            if 'category_real' in df.columns:
                total = len(df)
                accuracy = correct_count / total * 100
                print(f"\nEstadísticas de clasificación:")
                print(f"Total de registros: {total}")
                print(f"Correctamente clasificados: {correct_count} ({accuracy:.2f}%)")
                print(f"Incorrectamente clasificados: {total - correct_count} ({100 - accuracy:.2f}%)")

                # Guardar ejemplos incorrectos
                if incorrect_examples:
                    incorrect_df = pd.DataFrame(incorrect_examples)
                    incorrect_path = 'clasificaciones_incorrectas.csv'
                    incorrect_df.to_csv(incorrect_path, index=False)
                    print(f"\nEjemplos mal clasificados guardados en {incorrect_path}")

                    # Mostrar algunos ejemplos
                    print("\nAlgunos ejemplos de clasificaciones incorrectas:")
                    for i, example in enumerate(incorrect_examples[:5]):  # Mostrar hasta 5 ejemplos
                        print(f"Ejemplo {i+1}:")
                        print(f"  Texto: '{example['nombre']}'")
                        print(f"  Categoría real: {example['category_real']}")
                        print(f"  Categoría predicha: {example['category']}")

                    if len(incorrect_examples) > 5:
                        print(f"\n... y {len(incorrect_examples) - 5} ejemplos más en {incorrect_path}")


        except Exception as e:
            print(f"Error al procesar el archivo CSV: {e}")
    else:
        # Clasificar descripción individual
        description = ' '.join(sys.argv[1:])
        prediction = predict_room_type(description, model_data)
        print(f"Descripción: '{description}'")
        print(f"Clasificación: {prediction['category']} (ID: {prediction['category_id']})")

if __name__ == "__main__":
    main()
