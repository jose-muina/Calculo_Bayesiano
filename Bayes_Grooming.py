from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

data = {
    'text': [
        "Hola, ¿cómo estás? ¿Querés jugar algo online?",
       "¿Cuántos años tenés? No le digas a nadie que hablamos.",
        "Sos muy linda, no le muestres esto a tus papás.",
        "¿Terminaste la tarea de matemáticas?",
        "Podés mandarme una foto tuya, pero que sea solo para mí.",
        "¿Querés que te ayude con el juego que no podías pasar?",
        "No le contés a nadie que hablamos por acá, ¿sí?",
        "¿Te gustó la película de ayer?",
        "¿Estás solo en casa ahora?",
        "¿Querés hacer una videollamada para estudiar juntos?",
        "No le digas a tus padres que te escribí, es nuestro secreto.",
        "¿Cómo te fue en el examen de historia?",
        "Mandame una foto tuya, pero que nadie más la vea.",
        "¿Jugamos Minecraft esta tarde?",
        "Sos muy especial para mí, no le cuentes a nadie lo que hablamos.",
        "¿Querés que te pase los apuntes de biología?",
        "¿Podés mostrarme cómo estás vestida ahora?",
        "¿Tenés ganas de salir a andar en bici mañana?",
        "¿Querés que estudiemos juntos para el parcial?",
        "No hace falta que le digas a nadie que hablamos tanto."
    ],
    'label': [
        'no grooming', 'grooming', 'grooming', 'no grooming', 'grooming',
        'no grooming', 'grooming', 'no grooming', 'grooming', 'no grooming',
        'grooming', 'no grooming', 'grooming', 'no grooming', 'grooming',
        'no grooming', 'grooming', 'no grooming', 'no grooming', 'grooming'
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

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

nuevos_mensajes = [
    "¿Podes mandarme una foto tuya? No se la muestras a nadie",
    "¿Jugamos Minecaraft esta tarde?",
    "No le digas a tus papas que hablamos, ¿si?",
    "¿Terminaste el trabajo practico de historia?",
]

nuevos_vectorizados = vectorizer.transform(nuevos_mensajes)
predicciones = model.predict(nuevos_vectorizados)

for mensaje, prediccion in zip(nuevos_mensajes, predicciones):
    print(f'Mensaje: "{mensaje}" → Clasificación: {prediccion}')
