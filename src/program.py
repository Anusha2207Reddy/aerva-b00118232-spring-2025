import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

nlp = spacy.load("en_core_web_sm")

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("/Resume.csv")

# Encode target labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

# Check class distribution
df['Category'].value_counts().plot(kind='bar', title="Resume Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

df['processed_resume'] = df['Resume_str'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['processed_resume']).toarray()
y = df['Category_encoded'].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Naive Bayes 
class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}

    def fit(self, X, y):
        total_documents = len(y)
        unique_classes = np.unique(y)
        self.class_probs = {cls: np.sum(y == cls) / total_documents for cls in unique_classes}
        word_counts = {cls: np.zeros(X.shape[1]) for cls in unique_classes}
        class_word_totals = {cls: 0 for cls in unique_classes}
        
        for i in range(len(y)):
            label = y[i]
            word_counts[label] += X[i]
            class_word_totals[label] += np.sum(X[i])
        
        self.word_probs = {cls: (word_counts[cls] + 1) / (class_word_totals[cls] + X.shape[1]) for cls in unique_classes}

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            class_probs = {}
            for cls in self.class_probs:
                log_prob = np.log(self.class_probs[cls])
                word_probs = np.sum(np.log(self.word_probs[cls]) * X[i])
                class_probs[cls] = log_prob + word_probs
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb)}")
print(classification_report(y_test, y_pred_nb))

# Function to classify an individual resume
def classify_resume(resume_text):
    resume_tfidf = vectorizer.transform([resume_text])
    prediction = clf.predict(resume_tfidf)
    return prediction[0]

example_resume = "Experienced software developer with skills in Python and machine learning."
print("Predicted Category:", classify_resume(example_resume))


#Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)

lr = LogisticRegression(learning_rate=0.01, num_iterations=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(classification_report(y_test, y_pred_lr))

# 3. SVM with Hyperparameter Tuning
svm_param_grid = {'C': [0.1, 1, 10, 100], 'max_iter': [1000, 2000]}
svm_grid_search = GridSearchCV(LinearSVC(), svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)
svm_model = svm_grid_search.best_estimator_
y_pred_svm = svm_model.predict(X_test)
print(f"Best SVM Params: {svm_grid_search.best_params_}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(classification_report(y_test, y_pred_svm))

# 4. Random Forest with Hyperparameter Tuning
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
rf_model = rf_grid_search.best_estimator_
y_pred_rf = rf_model.predict(X_test)
print(f"Best RF Params: {rf_grid_search.best_params_}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Blues')
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Oranges')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Greens')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Purples')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance from Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,5))
plt.title("Feature Importance")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [tfidf_vectorizer.get_feature_names_out()[i] for i in indices[:10]], rotation=45)
plt.show()

# Feature Importance from SVM (using absolute weights)
svm_importances = np.abs(svm_model.coef_).mean(axis=0)
svm_indices = np.argsort(svm_importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importance (SVM)")
plt.bar(range(10), svm_importances[svm_indices[:10]], align="center")
plt.xticks(range(10), [tfidf_vectorizer.get_feature_names_out()[i] for i in svm_indices[:10]], rotation=45)
plt.show()

num_classes = df['Category'].nunique()
print(f"Number of unique classes: {num_classes}")


