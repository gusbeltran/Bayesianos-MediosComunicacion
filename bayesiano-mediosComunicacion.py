# Importar librerias necesarias
import pandas as pd

# Herramientas para procesamiento de texto
from sklearn.feature_extraction.text import CountVectorizer

# Modelo Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Metricas de evaluacion
from sklearn.metrics import accuracy_score, confusion_matrix


# CREACION DEL DATASET


datos = {
    "texto": [
        "El presidente anuncio una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Cientificos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cancer",
        "Gobierno aprueba ley de proteccion ambiental",
        "Investigadores aseguran que los telefonos espian nuestros sueños"
    ],
    "etiqueta": [
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake"
    ]
}

df = pd.DataFrame(datos)


# PREPARACION DE DATOS


X = df["texto"]
y = df["etiqueta"]

# Convertir texto en numeros
vectorizador = CountVectorizer()
X_vectorizado = vectorizador.fit_transform(X)


# ENTRENAMIENTO DEL MODELO


modelo = MultinomialNB()

# Entrenar con todo el conjunto
modelo.fit(X_vectorizado, y)

# Predicciones sobre el mismo conjunto
y_pred = modelo.predict(X_vectorizado)


# EVALUACION DEL MODELO


accuracy = accuracy_score(y, y_pred)
matriz = confusion_matrix(y, y_pred)

print("Accuracy:")
print(round(accuracy, 2))

print("\nMatriz de confusion:")
print(matriz)


# CLASIFICACION DE NUEVAS NOTICIAS

nuevas_noticias = [
    "Nuevo estudio demuestra que el cafe mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

nuevas_vectorizadas = vectorizador.transform(nuevas_noticias)

predicciones = modelo.predict(nuevas_vectorizadas)

print("\nClasificacion de nuevas noticias:")

for noticia, etiqueta in zip(nuevas_noticias, predicciones):
    print(f"{noticia} --> {etiqueta}")