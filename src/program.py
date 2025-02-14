import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("/content/sample_data/Resume.csv")

# Display dataset info
print(df.head())
print(df.info())

# Check class distribution
df['Category'].value_counts().plot(kind='bar', title="Resume Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

print(df.isnull().sum())

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Apply stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Apply preprocessing to the 'Resume_str' column
df['processed_resume'] = df['Resume_str'].apply(preprocess_text)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X = tfidf_vectorizer.fit_transform(df['processed_resume']).toarray()

# Assuming 'label' column has the target (0 for non-relevant, 1 for relevant)
y = df['Category'].values  # replace with actual label column name

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}

    def fit(self, X, y):
        # Calculate class probabilities
        total_documents = len(y)
        self.class_probs = {cls: np.sum(y == cls) / total_documents for cls in np.unique(y)}

        # Calculate word probabilities for each class
        word_counts = {cls: np.zeros(X.shape[1]) for cls in np.unique(y)}
        class_word_totals = {cls: 0 for cls in np.unique(y)}

        for i in range(len(y)):
            label = y[i]
            word_counts[label] += X[i]
            class_word_totals[label] += np.sum(X[i])

        self.word_probs = {cls: (word_counts[cls] + 1) / (class_word_totals[cls] + X.shape[1]) for cls in np.unique(y)}

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

# Initialize and train Naive Bayes
nb = NaiveBayes()
nb.fit(X, y)

most_frequent_class = df['Category_encoded'].value_counts().idxmax()

baseline_predictions = [most_frequent_class] * len(y)

baseline_accuracy = np.mean(baseline_predictions == y)
print(f"Baseline Accuracy: {baseline_accuracy}")

from sklearn.metrics import accuracy_score, classification_report

# Predict on the same data (or you can split data into train/test for better validation)
y_pred = nb.predict(X)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y, y_pred)}")
print(classification_report(y, y_pred))

print(df.columns)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression model
lr = LogisticRegression(max_iter=500)  # Increase max_iter to ensure convergence
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            # Calculate linear combination of features and weights
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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(learning_rate=0.01, num_iterations=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

from sklearn.svm import LinearSVC

# Initialize and train SVM model
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(classification_report(y_test, y_pred_svm))

from sklearn.ensemble import RandomForestClassifier

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))





