"""
Clasificador de Habitaciones de Hotel
=====================================
Este script implementa varios modelos para clasificar habitaciones de hotel
basados en sus descripciones textuales.
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Para preprocesamiento de texto
import string
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Para modelos de clasificación
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Modelos a probar
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


# Para procesamiento de lenguaje natural
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Descargar recursos de NLTK necesarios
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Definir el preprocesador de texto como clase global para permitir serialización
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_text(text) for text in X]

    def preprocess_text(self, text):
        """Preprocesa el texto para normalización"""
        # Convertir a mayúsculas
        text = text.upper()

        # Eliminar signos de puntuación
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()

        # Eliminar acentos
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')

        return text

class RoomClassifier:
    """Clase principal para clasificación de habitaciones de hotel"""

    def __init__(self):
        """Inicializa el clasificador con las categorías objetivo"""
        self.target_categories = {
            "APARTMENT": 0,
            "FAMILY ROOM": 1,
            "STUDIO": 2,
            "SHARED ROOM": 3,
            "SUITE": 4,
            "JUNIOR SUITE": 5,
            "PREMIUM ROOM": 6,
            "DELUXE ROOM": 7,
            "SUPERIOR ROOM": 8,
            "QUADRUPLE ROOM": 9,
            "TRIPLE ROOM": 10,
            "DOUBLE ROOM": 11,
            "STANDARD ROOM": 12,
            "CLASSIC ROOM":13

        }
        self.inv_categories = {v: k for k, v in self.target_categories.items()}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0

        # Stopwords en inglés y español
        self.stopwords = set(stopwords.words('english') + stopwords.words('spanish'))

        # Palabras clave para cada categoría (en inglés y español)
        #ESTOS ES IMPORTANTE PARA HACER LA CLASIFICACION INICIAL POR REGLAS, EL ORDEN IMPORTA PARA ESTABLECER PRIRORIDADES
        #CUANDO SE HAGA EL  ENTRENAMIENTO CONTINUO, ESTO YA NO IMPORTA
        self.category_keywords = {
            "APARTMENT": ["APARTMENT", "APARTAMENTO", "APTO"],
            "FAMILY ROOM": ["FAMILY", "FAMILIA", "FAM"],
            "STUDIO": ["STUDIO", "ESTUDIO", "STD"],
            "SHARED ROOM": ["SHARED", "COMPARTIDA", "COMPARTIDO","PARTILHADO","PARTILHADA"],
            "SUITE": ["SUITE"],
            "JUNIOR SUITE": ["JUNIOR SUITE", "JUNIOR", "JR SUITE", "JR. SUITE", "JSTE"],
            "PREMIUM ROOM": ["PREMIUM", "PREMIUM ROOM", "PRM","PREMIER"],
            "DELUXE ROOM": ["DELUXE", "DLX","LUXURY","LUJO"],
            "SUPERIOR ROOM": ["SUPERIOR", "SUP"],
            "QUADRUPLE ROOM": ["QUADRUPLE", "QUAD", "CUÁDRUPLE", "CUADRUPLE"],
            "TRIPLE ROOM": ["TRIPLE", "TPL"],
            "DOUBLE ROOM": ["DOUBLE", "DOBLE", "DBL"], #"TWIN"
            "STANDARD ROOM": ["STANDARD", "ESTÁNDAR", "ESTANDAR"],
            "CLASSIC ROOM": ["CLASSIC", "CLÁSICO", "CLASICO", "CLS"]

        }

        # Términos a ignorar (información de tarifa, etc.)
        self.ignore_terms = [
            "NONREFUNDABLE", "NO REEMBOLSABLE", "NO AMENDMENTS",
            "SIN CAMBIOS", "TARIFA", "RATE", "PAX", "ADULTS", "ADULTOS",
            "CHILD", "NIÑO", "NIÑOS", "CHILDREN", "EXTRA", "BED", "CAMA"
        ]

    def preprocess_text(self, text):
        """Preprocesa el texto para normalización"""
        # Convertir a mayúsculas
        text = text.upper()

        # Eliminar signos de puntuación
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()

        # Eliminar acentos
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')

        return text

    def extract_features(self, text):
        """Extrae características relevantes del texto"""
        processed_text = self.preprocess_text(text)

        # Características básicas
        features = {
            'text_length': len(text),
            'word_count': len(processed_text.split()),
            'has_bed_info': 1 if any(bed in processed_text for bed in
                                    ["KING", "QUEEN", "TWIN", "DOUBLE", "SINGLE"]) else 0,
            'has_rate_info': 1 if any(rate in processed_text for rate in
                                     ["NONREFUNDABLE", "NO REEMBOLSABLE"]) else 0
        }

        # Características de categoría
        for category, keywords in self.category_keywords.items():
            features[f'has_{category.lower().replace(" ", "_")}'] = 0
            for keyword in keywords:
                if keyword in processed_text:
                    features[f'has_{category.lower().replace(" ", "_")}'] = 1
                    break

        return features

    def rule_based_classifier(self, text):
        """Clasificador basado en reglas para establecer línea base"""
        processed_text = self.preprocess_text(text)

        # Eliminar términos a ignorar
        for term in self.ignore_terms:
            processed_text = processed_text.replace(term, "")

        # Buscar coincidencias con palabras clave de categorías
        matches = {}
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in processed_text:
                    matches[category] = matches.get(category, 0) + 1

        if not matches:
            # Si no hay coincidencias, intentar inferir por tipo de cama
            #if any(bed in processed_text for bed in ["KING", "QUEEN"]):
               # return self.target_categories.get("DELUXE ROOM")
            #elif "TWIN" in processed_text or "DOUBLE" in processed_text:
             #   return self.target_categories.get("DOUBLE ROOM")
           # else:
                return self.target_categories.get("STANDARD ROOM")

        # Devolver la categoría con más coincidencias
        best_category = max(matches.items(), key=lambda x: x[1])[0]
        return self.target_categories.get(best_category)

    def create_labeled_dataset(self, df):
        """Crea un conjunto de datos etiquetado para entrenamiento"""
        # Aplicar clasificador basado en reglas para generar etiquetas iniciales
        df['category_id'] = df['nombre'].apply(self.rule_based_classifier)
        df['category'] = df['category_id'].map(self.inv_categories)

        # Extraer características para cada descripción
        features_list = []
        for _, row in df.iterrows():
            features = self.extract_features(row['nombre'])
            features_list.append(features)

        features_df = pd.DataFrame(features_list)

        # Unir características con el DataFrame original
        df_features = pd.concat([df.reset_index(drop=True),
                                features_df.reset_index(drop=True)], axis=1)

        return df_features

    def train_models(self, X_train, y_train):
        """Entrena varios modelos y compara su rendimiento"""
        # Pipeline para TF-IDF con preprocesador global
        tfidf_pipeline = Pipeline([
            ('preprocessor', TextPreprocessor()),
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ])

        # Modelos a entrenar
        models = {
            'Naive Bayes': Pipeline([
                ('tfidf', tfidf_pipeline),
                ('classifier', MultinomialNB())
            ]),
            'Logistic Regression': Pipeline([
                ('tfidf', tfidf_pipeline),
                ('classifier', LogisticRegression(max_iter=1000, C=10))
            ]),
            'Random Forest': Pipeline([
                ('tfidf', tfidf_pipeline),
                ('classifier', RandomForestClassifier(n_estimators=100))
            ]),
            #'Linear SVM': Pipeline([
            #    ('tfidf', tfidf_pipeline),
             #   ('classifier', LinearSVC(C=1))
           # ]),
             'SGDClassifier (log=LogReg)': Pipeline([
            ('tfidf', tfidf_pipeline),
            ('classifier', SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3))
            ])
           #  'Passive-Aggressive': Pipeline([
            #('tfidf', tfidf_pipeline),
           # ('classifier', PassiveAggressiveClassifier(max_iter=1000, tol=1e-3))
           # ])
        }

        # Entrenar y evaluar cada modelo
        results = {}
        for name, model in models.items():
            print(f"Entrenando modelo: {name}")
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Guardar el modelo
            self.models[name] = model

            # Calcular precisión con validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            results[name] = {
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_time': train_time
            }

            print(f"  Precisión CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  Tiempo de entrenamiento: {train_time:.2f} segundos")

            # Actualizar mejor modelo
            if cv_scores.mean() > self.best_accuracy:
                self.best_accuracy = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name

        return results

    def evaluate_model(self, model, X_test, y_test):
        """Evalúa el rendimiento del modelo en datos de prueba"""
        y_pred = model.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)

        # Obtener las clases únicas presentes en los datos de prueba
        unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
        target_names = [self.inv_categories[i] for i in unique_classes]

        # Generar informe de clasificación solo con las clases presentes
        report = classification_report(y_test, y_pred,
                                      labels=unique_classes,
                                      target_names=target_names)

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'unique_classes': unique_classes,
            'target_names': target_names
        }

    def predict(self, text):

        """Predice la categoría de una descripción de habitación"""
        if self.best_model is None:
            # Si no hay modelo entrenado, usar clasificador basado en reglas
            category_id = self.rule_based_classifier(text)

        else:
            # Usar el mejor modelo entrenado
            category_id = self.best_model.predict([text])[0]

        return {
            'category_id': int(category_id),
            'category': self.inv_categories[int(category_id)],

        }

    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'target_categories': self.target_categories,
                'inv_categories': self.inv_categories,
                'category_keywords': self.category_keywords
            }, f)
        print(f"Modelo guardado en {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Carga un modelo guardado"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        classifier = cls()
        classifier.best_model = data['best_model']
        classifier.best_model_name = data['best_model_name']
        classifier.target_categories = data['target_categories']
        classifier.inv_categories = data['inv_categories']
        classifier.category_keywords = data['category_keywords']

        return classifier


def main():
    """Función principal para entrenar y evaluar modelos"""
    # Cargar datos
    print("Cargando datos...")
    df = pd.read_csv('sample_data_big.csv')


    # Crear clasificador
    classifier = RoomClassifier()

    # Crear conjunto de datos etiquetado
    print("Creando conjunto de datos etiquetado...")
    df_labeled = classifier.create_labeled_dataset(df)

    # Guardar conjunto etiquetado para referencia
    df_labeled.to_csv('labeled_data.csv', index=False)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        df_labeled['nombre'],
        df_labeled['category_id'],
        test_size=0.3,
        random_state=42
    )

    # Entrenar modelos
    print("\nEntrenando modelos...")
    results = classifier.train_models(X_train, y_train)

    # Evaluar mejor modelo
    print(f"\nEvaluando el mejor modelo ({classifier.best_model_name})...")
    evaluation = classifier.evaluate_model(classifier.best_model, X_test, y_test)

    print(f"Precisión en datos de prueba: {evaluation['accuracy']:.4f}")
    print("\nInforme de clasificación:")
    print(evaluation['classification_report'])

    # Guardar resultados
    with open('model_evaluation.txt', 'w') as f:
        f.write("EVALUACIÓN DE MODELOS DE CLASIFICACIÓN\n")
        f.write("=====================================\n\n")
        f.write("Resultados de validación cruzada:\n")
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Precisión CV: {result['cv_accuracy']:.4f} (±{result['cv_std']:.4f})\n")
            f.write(f"  Tiempo de entrenamiento: {result['train_time']:.2f} segundos\n\n")

        f.write(f"\nMejor modelo: {classifier.best_model_name}\n")
        f.write(f"Precisión en datos de prueba: {evaluation['accuracy']:.4f}\n\n")
        f.write("Informe de clasificación:\n")
        f.write(evaluation['classification_report'])

        f.write("\nNota: El informe de clasificación solo incluye las clases presentes en los datos de prueba.\n")
        f.write(f"Clases presentes: {len(evaluation['unique_classes'])} de {len(classifier.target_categories)}\n")

        f.write("\nObservaciones y recomendaciones:\n")
        f.write("1. La precisión actual está por debajo del objetivo del 98%. Esto se debe principalmente al tamaño limitado de los datos de muestra.\n")
        f.write("2. Para mejorar la precisión, se recomienda:\n")
        f.write("   - Aumentar el conjunto de datos de entrenamiento\n")
        f.write("   - Refinar las reglas de clasificación\n")
        f.write("   - Implementar técnicas de aumento de datos\n")
        f.write("   - Considerar modelos más avanzados de procesamiento de lenguaje natural\n")

    # Guardar modelo
    classifier.save_model('room_classifier_model.pkl')

    # Ejemplos de predicción
    print("\nEjemplos de predicción:")
    test_examples = [
        "DELUXE KING ROOM",
        "APARTAMENTO DE 2 DORMITORIOS",
        "STANDARD TWIN ROOM",
        "JUNIOR SUITE CON VISTAS AL MAR",
        "HABITACIÓN FAMILIAR PARA 4 PERSONAS"
    ]

    with open('prediction_examples.txt', 'w') as f:
        f.write("EJEMPLOS DE PREDICCIÓN\n")
        f.write("=====================\n\n")

        for example in test_examples:
            prediction = classifier.predict(example)
            print(f"Texto: '{example}'")
            print(f"Predicción: {prediction['category']} (ID: {prediction['category_id']})")
            print()

            f.write(f"Texto: '{example}'\n")
            f.write(f"Predicción: {prediction['category']} (ID: {prediction['category_id']})\n\n")

    # Removero de desarrollo: eliminar el archivo de datos etiquetados
    # Mostrar ejemplos mal clasificados
    print("\nEjemplos mal clasificados:")
    y_pred = classifier.best_model.predict(X_test)
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        if pred != true:
            true_category = classifier.inv_categories[int(true)]
            pred_category = classifier.inv_categories[int(pred)]
            print(f"Texto: '{X_test.iloc[i]}'")
            print(f"  Categoría real: {true_category} (ID: {true})")
            print(f"  Categoría predicha: {pred_category} (ID: {pred})")
            print()

    # Guardar ejemplos mal clasificados en archivo
    with open('misclassified_examples.txt', 'w') as f:
        f.write("EJEMPLOS MAL CLASIFICADOS\n")
        f.write("=======================\n\n")
        for i, (pred, true) in enumerate(zip(y_pred, y_test)):
            if pred != true:
                true_category = classifier.inv_categories[int(true)]
                pred_category = classifier.inv_categories[int(pred)]
                f.write(f"Texto: '{X_test.iloc[i]}'\n")
                f.write(f"  Categoría real: {true_category} (ID: {true})\n")
                f.write(f"  Categoría predicha: {pred_category} (ID: {pred})\n\n")
    # fin mal clasificados


    # Crear visualización de matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(evaluation['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=evaluation['target_names'],
               yticklabels=evaluation['target_names'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    print("Evaluación completa. Resultados guardados en 'model_evaluation.txt'")
    print("Matriz de confusión guardada en 'confusion_matrix.png'")
    print("Ejemplos de predicción guardados en 'prediction_examples.txt'")


if __name__ == "__main__":
    main()
