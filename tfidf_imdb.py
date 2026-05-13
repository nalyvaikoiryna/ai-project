# ДРУГИЙ ЕКСПЕРИМЕНТ:
# IMDb sentiment classification
# TF-IDF + Logistic Regression
# Дані беруться з одного CSV-файлу і діляться на train/test

import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. ОЧИЩЕННЯ ТЕКСТУ
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)         # прибрати HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # залишити тільки літери
    text = re.sub(r"\s+", " ", text).strip()   # прибрати зайві пробіли
    return text


# 2. ЗАВАНТАЖЕННЯ ДАНИХ
df = pd.read_csv("IMDB Dataset.csv")

print("Перші 5 рядків:")
print(df.head())

print("\nНазви колонок:")
print(df.columns)

print("\nРозмір датасету:", df.shape)


# 3. ПОПЕРЕДНЯ ОБРОБКА
df["review"] = df["review"].apply(clean_text)

# перетворення міток у числа
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# видалення порожніх значень
df = df.dropna(subset=["review", "sentiment"])


# 4. ОЗНАКИ І МІТКИ
X = df["review"]
y = df["sentiment"]


# 5. ПОДІЛ НА TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. TF-IDF ВЕКТОРИЗАЦІЯ
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 7. НАВЧАННЯ МОДЕЛІ
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# 8. ПРОГНОЗ
y_pred = model.predict(X_test_tfidf)


# 9. ОЦІНКА ЯКОСТІ
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))