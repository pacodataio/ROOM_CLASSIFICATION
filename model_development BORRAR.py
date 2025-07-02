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
from sklearn.linear_model import LogisticRegression
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
            "SUITE": 0,
            "JUNIOR SUITE": 1,
            "PREMIUM ROOM": 2,
            "DELUXE ROOM": 3,
            "SUPERIOR ROOM": 4,
            "QUADRUPLE ROOM": 5,
            "TRIPLE ROOM": 6,
            "DOUBLE ROOM": 7,
            "STANDARD ROOM": 8,
            "CLASSIC ROOM":9,
            "APARTMENT": 10,
            "STUDIO": 11,
            "SHARED ROOM": 12,
            "FAMILY ROOM": 13
        }
        self.inv_categories = {v: k for k, v in self.target_categories.items()}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0

        # Stopwords en inglés y español
        self.stopwords = set(stopwords.words('english') + stopwords.words('spanish'))

        # Palabras clave para cada categoría (en inglés y español)
        self.category_keywords = {
            "SUITE": ["SUITE", "STE"],
            "JUNIOR SUITE": ["JUNIOR SUITE", "JUNIOR", "JR SUITE", "JR. SUITE", "JSTE"],
            "PREMIUM ROOM": ["PREMIUM", "PREMIUM ROOM", "PRM"],
            "DELUXE ROOM": ["DELUXE", "DLX"],
            "SUPERIOR ROOM": ["SUPERIOR", "SUP"],
            "QUADRUPLE ROOM": ["QUADRUPLE", "QUAD", "CUÁDRUPLE", "CUADRUPLE"],
            "TRIPLE ROOM": ["TRIPLE", "TPL"],
            "DOUBLE ROOM": ["DOUBLE", "DOBLE", "DBL"], #"TWIN"
            "STANDARD ROOM": ["STANDARD", "ESTÁNDAR", "ESTANDAR"],
            "CLASSIC ROOM": ["CLASSIC", "CLÁSICO", "CLASICO", "CLS"],
            "APARTMENT": ["APARTMENT", "APARTAMENTO", "APTO", "APT"],
            "STUDIO": ["STUDIO", "ESTUDIO", "STD"],
            "SHARED ROOM": ["SHARED", "COMPARTIDA", "COMPARTIDO"],
            "FAMILY ROOM": ["FAMILY", "FAMILIA", "FAM"]
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
            'Linear SVM': Pipeline([
                ('tfidf', tfidf_pipeline),
                ('classifier', LinearSVC(C=1))
            ])
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
            'category': self.inv_categories[int(category_id)]
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
    df = pd.read_csv('del.csv')


    # Crear clasificador
    classifier = RoomClassifier()

    # Crear conjunto de datos etiquetado
    print("Creando conjunto de datos etiquetado...")
    df_labeled = classifier.create_labeled_dataset(df)

    # Guardar conjunto etiquetado para referencia
    df_labeled.to_csv('labeled_data.csv', index=False)


if __name__ == "__main__":
    main()
