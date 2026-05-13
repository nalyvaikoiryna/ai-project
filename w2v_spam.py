# SMS Spam Collection / spam.csv
# Word2Vec + Logistic Regression

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
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. ТОКЕНІЗАЦІЯ
def tokenize(text):
    return text.split()


# 3. СЕРЕДНІЙ ВЕКТОР ДОКУМЕНТА
def document_vector(tokens, model, vector_size):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if len(valid_vectors) == 0:
        return np.zeros(vector_size)
    
    return np.mean(valid_vectors, axis=0)


# 4. ЗАВАНТАЖЕННЯ ДАНИХ
df = pd.read_csv(
    "/Users/macbook/shi/spam.csv",
    encoding="latin-1"
)

print("Перші 5 рядків:")
print(df.head())

print("\nНазви колонок:")
print(df.columns)

print("\nРозмір датасету:", df.shape)


# 5. ЗАЛИШАЄМО ТІЛЬКИ ПОТРІБНІ КОЛОНКИ
df = df[["v1", "v2"]].copy()
df.columns = ["label", "message"]

print("\nКласи:")
print(df["label"].value_counts())


# 6. ПОПЕРЕДНЯ ОБРОБКА
df["message"] = df["message"].apply(clean_text)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

df = df.dropna(subset=["label", "message"])


# 7. ОЗНАКИ І МІТКИ
X = df["message"]
y = df["label"]


# 8. ПОДІЛ НА TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 9. ТОКЕНІЗАЦІЯ
X_train_tokens = X_train.apply(tokenize)
X_test_tokens = X_test.apply(tokenize)


# 10. НАВЧАННЯ WORD2VEC НА TRAIN
vector_size = 100

w2v_model = Word2Vec(
    sentences=X_train_tokens,
    vector_size=vector_size,
    window=5,
    min_count=1,
    workers=4,
    sg=0
)


# 11. ПЕРЕТВОРЕННЯ ДОКУМЕНТІВ У ВЕКТОРИ
X_train_w2v = np.array([
    document_vector(tokens, w2v_model, vector_size)
    for tokens in X_train_tokens
])

X_test_w2v = np.array([
    document_vector(tokens, w2v_model, vector_size)
    for tokens in X_test_tokens
])


# 12. НАВЧАННЯ КЛАСИФІКАТОРА
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_w2v, y_train)


# 13. ПРОГНОЗ
y_pred = model.predict(X_test_w2v)


# 14. ОЦІНКА ЯКОСТІ
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))