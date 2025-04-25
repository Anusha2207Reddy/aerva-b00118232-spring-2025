Automated Resume Classification using Machine Learning

This project implements a multi-class text classification system to automatically categorize resumes into different job categories. It includes manual implementations of Naive Bayes and Logistic Regression classifiers, and comparisons with powerful models like SVM and Random Forest, using TF-IDF vectorization for feature extraction.

Dataset

The dataset used is Resume.csv, which contains:
Resume_str: Raw text of the resume
Category: Target label indicating the resume's category
Note: Make sure the dataset is placed in the root directory or update the path accordingly in the script.

Dependencies

To run this project, you need to install the following Python libraries:

numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
spacy
