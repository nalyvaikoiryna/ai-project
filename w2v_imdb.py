# ТРЕТІЙ ЕКСПЕРИМЕНТ:
# IMDb sentiment classification
# Word2Vec + Logistic Regression
# Дані беруться з одного CSV-файлу і діляться на train/test

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from gensim.models import Word2Vec


# 1. ОЧИЩЕННЯ ТЕКСТУ
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)         # прибрати HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # залишити тільки літери
    text = re.sub(r"\s+", " ", text).strip()   # прибрати зайві пробіли
    return text


# 2. ТОКЕНІЗАЦІЯ
def tokenize(text):
    return text.split()


# 3. ФУНКЦІЯ ДЛЯ СЕРЕДНЬОГО ВЕКТОРА ДОКУМЕНТА
def document_vector(tokens, model, vector_size):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]

    if len(valid_vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(valid_vectors, axis=0)


# 4. ЗАВАНТАЖЕННЯ ДАНИХ
df = pd.read_csv("IMDB Dataset.csv")

print("Перші 5 рядків:")
print(df.head())

print("\nНазви колонок:")
print(df.columns)

print("\nРозмір датасету:", df.shape)


# 5. ПОПЕРЕДНЯ ОБРОБКА
df["review"] = df["review"].apply(clean_text)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
df = df.dropna(subset=["review", "sentiment"])


# 6. ОЗНАКИ І МІТКИ
X = df["review"]
y = df["sentiment"]


# 7. ПОДІЛ НА TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 8. ТОКЕНІЗАЦІЯ ТРЕНУВАЛЬНИХ І ТЕСТОВИХ ТЕКСТІВ
X_train_tokens = X_train.apply(tokenize)
X_test_tokens = X_test.apply(tokenize)


# 9. НАВЧАННЯ WORD2VEC ТІЛЬКИ НА TRAIN
vector_size = 100

w2v_model = Word2Vec(
    sentences=X_train_tokens,
    vector_size=vector_size,
    window=5,
    min_count=2,
    workers=4,
    sg=0
)


# 10. ПЕРЕТВОРЕННЯ КОЖНОГО ДОКУМЕНТА В ОДИН ВЕКТОР
X_train_w2v = np.array([
    document_vector(tokens, w2v_model, vector_size)
    for tokens in X_train_tokens
])

X_test_w2v = np.array([
    document_vector(tokens, w2v_model, vector_size)
    for tokens in X_test_tokens
])


# 11. НАВЧАННЯ КЛАСИФІКАТОРА
model = LogisticRegression(max_iter=1000)
model.fit(X_train_w2v, y_train)


# 12. ПРОГНОЗ
y_pred = model.predict(X_test_w2v)


# 13. ОЦІНКА ЯКОСТІ
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))