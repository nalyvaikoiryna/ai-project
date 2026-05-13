# 20 Newsgroups
# Word2Vec + Logistic Regression

import re
import numpy as np

from sklearn.datasets import fetch_20newsgroups
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


# 4. ВИБІР КАТЕГОРІЙ
categories = [
    "rec.sport.hockey",
    "sci.space",
    "talk.politics.misc",
    "comp.graphics"
]


# 5. ЗАВАНТАЖЕННЯ TRAIN І TEST
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


# 6. ТОКЕНІЗАЦІЯ
X_train_tokens = [tokenize(text) for text in X_train]
X_test_tokens = [tokenize(text) for text in X_test]


# 7. НАВЧАННЯ WORD2VEC НА TRAIN
vector_size = 100

w2v_model = Word2Vec(
    sentences=X_train_tokens,
    vector_size=vector_size,
    window=5,
    min_count=2,
    workers=4,
    sg=0
)


# 8. ПЕРЕТВОРЕННЯ ДОКУМЕНТІВ У ВЕКТОРИ
X_train_w2v = np.array([
    document_vector(tokens, w2v_model, vector_size)
    for tokens in X_train_tokens
])

X_test_w2v = np.array([
    document_vector(tokens, w2v_model, vector_size)
    for tokens in X_test_tokens
])


# 9. НАВЧАННЯ КЛАСИФІКАТОРА
model = LogisticRegression(max_iter=2000)
model.fit(X_train_w2v, y_train)


# 10. ПРОГНОЗ
y_pred = model.predict(X_test_w2v)


# 11. ОЦІНКА ЯКОСТІ
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=train_data.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))