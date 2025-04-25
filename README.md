# 🧠 Resume Classification using NLP and Machine Learning

This project implements a multi-class **Resume Classification** system that classifies resumes into categories such as HR, Data Science, Software Developer, etc., using classical NLP and Machine Learning techniques.

## 📌 Features
- Preprocessing resumes using NLTK and SpaCy
- TF-IDF vectorization of text
- Custom implementation of:
  - Naive Bayes Classifier
  - Logistic Regression Classifier
- Scikit-learn models:
  - Support Vector Machine (SVM) with hyperparameter tuning
  - Random Forest Classifier with hyperparameter tuning
- Visualizations:
  - Confusion matrices
  - Feature importance graphs
- Resume classification function for predicting new examples

## 📁 Dataset
- Source: [Kaggle - Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- File: `Resume.csv`
- Columns: 
  - `Resume_str`: The resume text
  - `Category`: The job field/category label

## ⚙️ Dependencies

Install all the dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn spacy
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

## 📋 Project Structure

```
resume-classifier/
├── Resume.csv
├── resume_classifier.py        # Main script containing code
├── README.md                   # Project overview and instructions
└── visuals/                    # Optional: save confusion matrices and plots here
```

## 🧪 How to Run

1. **Place `Resume.csv`** in the project directory.
2. **Run the main script**:

```bash
python resume_classifier.py
```

3. The script:
   - Loads and preprocesses the data
   - Trains Naive Bayes, Logistic Regression, SVM, and Random Forest classifiers
   - Prints accuracy and classification reports
   - Plots confusion matrices and feature importances
   - Classifies a sample resume using the best-performing model (SVM)

## 🧠 Models Implemented

| Model               | Custom Implementation | Accuracy Measured | Hyperparameter Tuning |
|--------------------|-----------------------|-------------------|------------------------|
| Naive Bayes        | ✅                    | ✅                | ❌                     |
| Logistic Regression| ✅                    | ✅                | ❌                     |
| SVM                | ❌ (sklearn)          | ✅                | ✅                     |
| Random Forest      | ❌ (sklearn)          | ✅                | ✅                     |

## 📊 Sample Output

```
Naive Bayes Accuracy: 0.87
Logistic Regression Accuracy: 0.63
SVM Accuracy: 0.92
Random Forest Accuracy: 0.91
```

## 🧪 Classify a New Resume

You can use the following function in your code to classify a custom resume:

```python
def classify_resume(resume_text):
    resume_tfidf = tfidf_vectorizer.transform([resume_text])
    prediction = svm_model.predict(resume_tfidf)
    return le.inverse_transform([prediction[0]])[0]

# Example usage
resume = "Experienced software developer with skills in Python and machine learning."
print("Predicted Category:", classify_resume(resume))
```

## 📈 Visualizations

- Resume category distribution (bar chart)
- Confusion matrices for all models
- Top 10 important features (TF-IDF terms) for SVM and Random Forest

## 📜 License

This project is for educational and research purposes only. Not intended for real-world recruitment applications.
