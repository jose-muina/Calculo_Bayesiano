from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

data = {
    'text': [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte ",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños",
    ],
    'label': [
        'real', 'fake', 'real', 'fake', 'real',
        'fake', 'real', 'fake', 'real', 'fake',
    ]
}

df = pd.DataFrame(data)

X = df['text']
y = df['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos",
]

nuevas_vectorizadas = vectorizer.transform(nuevas_noticias)
predicciones = model.predict(nuevas_vectorizadas)

for noticia, prediccion in zip(nuevas_noticias, predicciones):
    print(f'Noticia: "{noticia}" → Clasificación: {prediccion}')

for mensaje, prediccion in zip(nuevos_mensajes, predicciones):
    print(f'Mensaje: "{mensaje}" → Clasificación: {prediccion}')
