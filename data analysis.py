import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

# Cargar los datos
df = pd.read_csv('sample_data_big.csv')

# Mostrar información básica
print('Información del DataFrame:')
print(df.info())
print('\nPrimeras 5 filas:')
print(df.head())

# Análisis de idiomas y patrones
print('\nAnálisis de patrones en las descripciones:')
# Detectar idioma (simplificado)
def detect_language(text):
    spanish_words = ['días', 'noches', 'reembolsable', 'empaquetada']
    for word in spanish_words:
        if word.lower() in text.lower():
            return 'Spanish'
    return 'English'

df['language'] = df['nombre'].apply(detect_language)

print(f'Distribución de idiomas: {df.language.value_counts().to_dict()}')
print(df)
# Análisis de palabras clave
def extract_keywords(text):
    text = text.upper()
    room_types = ['APARTMENT', 'STANDARD', 'DOUBLE', 'TRIPLE', 'CLASSIC',
                 'SUPERIOR', 'DELUXE', 'PREMIUM', 'JUNIOR SUITE', 'SUITE',
                 'QUADRUPLE', 'STUDIO', 'SHARED', 'FAMILY']
    found = []
    for room_type in room_types:
        if room_type in text:
            found.append(room_type)
    return found if found else ['UNKNOWN']

df['keywords'] = df['nombre'].apply(extract_keywords)
keywords_flat = [item for sublist in df['keywords'].tolist() for item in sublist]

print(f'\nDistribución de palabras clave: {Counter(keywords_flat)}')
print(df)
# Análisis de longitud de descripciones
df['desc_length'] = df['nombre'].apply(len)
print(f'\nEstadísticas de longitud de descripciones:')
print(df['desc_length'].describe())

# Análisis de patrones adicionales
def has_rate_info(text):
    return 'NONREFUNDABLE' in text.upper() or 'NO REEMBOLSABLE' in text.upper()

def has_bed_info(text):
    return bool(re.search(r'(KING|QUEEN|TWIN|DOUBLE|SINGLE)\s+(BED|BEDS)?', text.upper()))

df['has_rate_info'] = df['nombre'].apply(has_rate_info)
df['has_bed_info'] = df['nombre'].apply(has_bed_info)

print(f'\nDescripciones con información de tarifa: {df.has_rate_info.sum()} ({df.has_rate_info.mean()*100:.1f}%)')
print(f'Descripciones con información de cama: {df.has_bed_info.sum()} ({df.has_bed_info.mean()*100:.1f}%)')

# Guardar resultados del análisis
with open('data_analysis.txt', 'w') as f:
    f.write('ANÁLISIS DE DATOS DE HABITACIONES DE HOTEL\n')
    f.write('=========================================\n\n')
    f.write(f'Total de registros analizados: {len(df)}\n')
    f.write(f'Distribución de idiomas: {df.language.value_counts().to_dict()}\n\n')
    f.write(f'Distribución de palabras clave: {Counter(keywords_flat)}\n\n')
    f.write(f'Estadísticas de longitud de descripciones:\n{df.desc_length.describe()}\n\n')
    f.write(f'Descripciones con información de tarifa: {df.has_rate_info.sum()} ({df.has_rate_info.mean()*100:.1f}%)\n')
    f.write(f'Descripciones con información de cama: {df.has_bed_info.sum()} ({df.has_bed_info.mean()*100:.1f}%)\n')

print('\nAnálisis guardado en data_analysis.txt')

# Crear visualización de distribución de palabras clave
plt.figure(figsize=(12, 6))
counter = Counter(keywords_flat)
labels = [x for x in counter.keys() if x != 'UNKNOWN']
values = [counter[x] for x in labels]
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.title(f'Distribución de Tipos de Habitación ({len(df)} datos analizados))')
plt.tight_layout()
plt.savefig('keyword_distribution.png')
print('Visualización guardada en keyword_distribution.png')