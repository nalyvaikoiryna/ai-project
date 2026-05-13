# 20 Newsgroups
# Bag of Words + Logistic Regression

import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. ОЧИЩЕННЯ ТЕКСТУ
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # залишаємо тільки літери
    text = re.sub(r"\s+", " ", text).strip()   # прибираємо зайві пробіли
    return text


# 2. ВИБІР КАТЕГОРІЙ
# Можна брати всі 20, але для курсової часто зручніше 4 категорії
# categories = [
#     "rec.sport.hockey",
#     "sci.space",
#     "talk.politics.misc",
#     "comp.graphics"
# ]


# 3. ЗАВАНТАЖЕННЯ TRAIN І TEST
train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

X_train = [clean_text(text) for text in train_data.data]
X_test = [clean_text(text) for text in test_data.data]
y_train = train_data.target
y_test = test_data.target


# 4. BAG OF WORDS
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)


# 5. НАВЧАННЯ МОДЕЛІ
model = LogisticRegression(max_iter=2000)
model.fit(X_train_bow, y_train)


# 6. ПРОГНОЗ
y_pred = model.predict(X_test_bow)


# 7. ОЦІНКА ЯКОСТІ
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=train_data.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))