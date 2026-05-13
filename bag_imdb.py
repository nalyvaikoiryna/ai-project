# ПЕРШИЙ ЕКСПЕРИМЕНТ:
# IMDb sentiment classification
# Bag of Words + Logistic Regression
# Дані беруться з одного CSV-файлу і діляться на train/test

import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
# positive -> 1
# negative -> 0
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# якщо є порожні значення — видаляємо
df = df.dropna(subset=["review", "sentiment"])


# 4. ОЗНАКИ І МІТКИ
X = df["review"]
y = df["sentiment"]


# 5. ПОДІЛ НА TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42,
    stratify=y          # зберігає баланс класів
)


# 6. BAG OF WORDS
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)


# 7. НАВЧАННЯ МОДЕЛІ
model = LogisticRegression(max_iter=10000)
model.fit(X_train_bow, y_train)


# 8. ПРОГНОЗ
y_pred = model.predict(X_test_bow)


# 9. ОЦІНКА ЯКОСТІ
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))