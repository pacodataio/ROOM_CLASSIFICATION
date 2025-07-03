from transformers import pipeline

# Carga un clasificador preentrenado (multilingüe)
classifier = pipeline("text-classification", model="Suchinthana/Finetuned-BERT-HotelClassification")

# Tu texto a clasificar
text = "habitacion con vista al mar, ideal para una escapada romántica."

# Obtener la predicción
result = classifier(text)

print(result)
