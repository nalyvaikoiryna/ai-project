# SMS Spam Collection
# Bag of Words + Logistic Regression

import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. ОЧИЩЕННЯ ТЕКСТУ
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. ЗАВАНТАЖЕННЯ ДАНИХ
df = pd.read_csv(
    "/Users/macbook/shi/spam.csv",
    encoding="latin-1"
)

print("Перші 5 рядків:")
print(df.head())

print("\nНазви колонок:")
print(df.columns)

print("\nРозмір датасету:", df.shape)


# 3. ЗАЛИШАЄМО ТІЛЬКИ ПОТРІБНІ КОЛОНКИ
df = df[["v1", "v2"]].copy()
df.columns = ["label", "message"]

print("\nКласи:")
print(df["label"].value_counts())


# 4. ПОПЕРЕДНЯ ОБРОБКА
df["message"] = df["message"].apply(clean_text)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

df = df.dropna(subset=["label", "message"])


# 5. ОЗНАКИ І МІТКИ
X = df["message"]
y = df["label"]


# 6. ПОДІЛ НА TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. BAG OF WORDS
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)


# 8. НАВЧАННЯ МОДЕЛІ
model = LogisticRegression(max_iter=1000)
model.fit(X_train_bow, y_train)


# 9. ПРОГНОЗ
y_pred = model.predict(X_test_bow)


# 10. ОЦІНКА ЯКОСТІ
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))