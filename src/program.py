import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("/Users/anushareddy/Desktop/aerva-b00118232-spring-2025/data/Resume.csv")  

# Display dataset info
print(df.head())
print(df.info())

# Check class distribution
df['Category'].value_counts().plot(kind='bar', title="Resume Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()
print(df.isnull().sum())

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters & numbers
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return ' '.join(tokens)

# Apply preprocessing to Resume_str column
df['Processed_Resume'] = df['Resume_str'].apply(preprocess_text)

# Build vocabulary
def build_vocab(texts):
    vocab = set()
    for text in texts:
        vocab.update(text.split())
    return list(vocab)

# Convert text to vector
def text_to_vector(text, vocab):
    word_count = Counter(text.split())
    return np.array([word_count[word] if word in word_count else 0 for word in vocab])

# Build vocabulary from preprocessed text
vocab = build_vocab(df['Processed_Resume'])

# Convert each resume to a vector
X = np.array([text_to_vector(text, vocab) for text in df['Processed_Resume']])

# Encode labels as numerical values
categories = {cat: idx for idx, cat in enumerate(df['Category'].unique())}
y = df['Category'].map(categories).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}  # P(Class)
        self.word_probs = defaultdict(lambda: defaultdict(float))  
        self.vocab_size = 0

    def fit(self, X, y):
        num_docs = len(y)
        num_classes = len(set(y))
        class_counts = Counter(y)

        # Calculate P(Class)
        self.class_probs = {cls: class_counts[cls] / num_docs for cls in class_counts}

        # Calculate P(Word | Class)
        word_counts = {cls: np.zeros(X.shape[1]) for cls in class_counts}
        for i in range(len(X)):
            word_counts[y[i]] += X[i]

        # Laplace smoothing
        self.vocab_size = X.shape[1]
        for cls in class_counts:
            self.word_probs[cls] = (word_counts[cls] + 1) / (sum(word_counts[cls]) + self.vocab_size)

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for cls in self.class_probs:
                log_prob = np.log(self.class_probs[cls]) + np.sum(np.log(self.word_probs[cls]) * x)
                class_scores[cls] = log_prob
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

# Train model
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=categories.keys()))

