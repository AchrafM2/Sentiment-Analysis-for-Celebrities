import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats as stats

import numpy as np

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer

import re

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import train_test_split

import nltk

from nltk.corpus import stopwords

import string
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd


df=pd.read_csv("C:/Users/ACHRAF/Desktop/SaadLamjarred_LV.CSV")
pattern = r'[^\w\s-]'
df = df.applymap(lambda x: re.sub(pattern, '', str(x)))
nltk.download('stopwords')
df
# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Function to apply stemming to a text
def apply_stemming(text):
    tokens = word_tokenize(text)  # Tokenize the text into individual words
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming to each word
    stemmed_text = ' '.join(stemmed_tokens)  # Join the stemmed tokens back into a string
    return stemmed_text

# Apply stemming to all columns in the dataframe
df = df.applymap(apply_stemming)
df
import nltk

nltk.download('punkt')
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []

    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    
    return stems
df=df[['Comments','Labels']] 
X=df['Comments']
arabic_stopwords = stopwords.words('arabic')

token_dict = {}

for i in range(884):
    token_dict[i]=X[i]
    
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=arabic_stopwords)
tfs = tfidf.fit_transform(token_dict.values())

Xf=pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names())
# Split the data into train, test, and validation sets
X_train_val, X_test, y_train_val, y_test = train_test_split(df['Comments'], df['Labels'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Convert the text to numerical features using TF-IDF
X_train_tfidf = np.where(pd.isnull(X_train), '', X_train)
X_val_tfidf = np.where(pd.isnull(X_val), '', X_val)
X_test_tfidf = np.where(pd.isnull(X_test), '', X_test)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train.fillna(''))
X_val_tfidf = tfidf.transform(X_val.fillna(''))
X_test_tfidf = tfidf.transform(X_test.fillna(''))

# Train and evaluate logistic regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_tfidf, y_train)
print("Logistic Regression")
print("Training accuracy:", lr.score(X_train_tfidf, y_train))
print("Validation accuracy:", lr.score(X_val_tfidf, y_val))
print("Test accuracy:", lr.score(X_test_tfidf, y_test))
print()


nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
print("Naive Bayes")
print("Training accuracy:", nb.score(X_train_tfidf, y_train))
print("Validation accuracy:", nb.score(X_val_tfidf, y_val))
print("Test accuracy:", nb.score(X_test_tfidf, y_test))
print()


# Train and evaluate LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_tfidf.toarray(), y_train)
print("LDA")
print("Training accuracy:", lda.score(X_train_tfidf.toarray(), y_train))
print("Validation accuracy:", lda.score(X_val_tfidf.toarray(), y_val))
print("Test accuracy:", lda.score(X_test_tfidf.toarray(), y_test))
print()

# Train and evaluate QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_tfidf.toarray(), y_train)
print("QDA")
print("Training accuracy:", qda.score(X_train_tfidf.toarray(), y_train))
print("Validation accuracy:", qda.score(X_val_tfidf.toarray(), y_val))
print("Test accuracy:", qda.score(X_test_tfidf.toarray(), y_test))
print()


# Use grid search to optimize hyperparameters for logistic regression
lr = LogisticRegression(random_state=42)
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
print("Logistic Regression with Grid Search")
print("Training accuracy:", grid_search.score(X_train_tfidf, y_train))
print("Validation accuracy:", grid_search.score(X_val_tfidf, y_val))
print("Test accuracy:", grid_search.score(X_test_tfidf, y_test))
print("Best parameters:", grid_search.best_params_)
print()


new_comment_tfidf = tfidf.transform(["أعانه الله ووفقه لما فيه الخير وفرج همه يارب ان شاء لله نراه خارج اسوار السجن بصفه نهائية"])
predicted_label = lr.predict(new_comment_tfidf)[0]

# Print the predicted sentiment label
print("Predicted sentiment label:", predicted_label)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

# Define the QDA model and the parameter grid
qda = QuadraticDiscriminantAnalysis()
param_grid = {'reg_param': [0.01, 0.1, 1.0]}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(qda, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the results
print("QDA with Grid Search")
print("Training accuracy:", grid_search.score(X_train, y_train))
print("Validation accuracy:", grid_search.score(X_val, y_val))
print("Test accuracy:", grid_search.score(X_test, y_test))
print("Best parameters:", grid_search.best_params_)
print()

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# Define the Naive Bayes model and the parameter grid
nb = MultinomialNB()
param_grid = {'alpha': [0.01, 0.1, 1.0], 'fit_prior': [True, False]}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(nb, param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Print the results
print("Naive Bayes with Grid Search")
print("Training accuracy:", grid_search.score(X_train_tfidf, y_train))
print("Validation accuracy:", grid_search.score(X_val_tfidf, y_val))
print("Test accuracy:", grid_search.score(X_test_tfidf, y_test))
print("Best parameters:", grid_search.best_params_)
print()

from sklearn.ensemble import RandomForestClassifier

# Train and evaluate Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_tfidf, y_train)
print("Random Forest")
print("Training accuracy:", rf.score(X_train_tfidf, y_train))
print("Validation accuracy:", rf.score(X_val_tfidf, y_val))
print("Test accuracy:", rf.score(X_test_tfidf, y_test))
print()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the Random Forest model and the parameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Print the results
print("Random Forest with Grid Search")
print("Training accuracy:", grid_search.score(X_train_tfidf, y_train))
print("Validation accuracy:", grid_search.score(X_val_tfidf, y_val))
print("Test accuracy:", grid_search.score(X_test_tfidf, y_test))
print("Best parameters:", grid_search.best_params_)
print()


from sklearn.neighbors import KNeighborsClassifier

# Train and evaluate KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)
print("KNN")
print("Training accuracy:", knn.score(X_train_tfidf, y_train))
print("Validation accuracy:", knn.score(X_val_tfidf, y_val))
print("Test accuracy:", knn.score(X_test_tfidf, y_test))
print()


from sklearn.svm import SVC

# Train and evaluate SVM
svm = SVC()
svm.fit(X_train_tfidf, y_train)
print("SVM")
print("Training accuracy:", svm.score(X_train_tfidf, y_train))
print("Validation accuracy:", svm.score(X_val_tfidf, y_val))
print("Test accuracy:", svm.score(X_test_tfidf, y_test))


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

# Create a QDA classifier
qda = QuadraticDiscriminantAnalysis()

# Define the parameter grid
param_grid = {'reg_param': [0.1, 0.5, 1.0]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(qda, param_grid, cv=5)
grid_search.fit(X_train_tfidf.toarray(), y_train)

# Print the results
print("QDA with Grid Search")
print("Training accuracy:", grid_search.score(X_train_tfidf.toarray(), y_train))
print("Validation accuracy:", grid_search.score(X_val_tfidf.toarray(), y_val))
print("Test accuracy:", grid_search.score(X_test_tfidf.toarray(), y_test))
print("Best parameters:", grid_search.best_params_)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Define the KNN model and the parameter grid
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Print the results
print("KNN with Grid Search")
print("Training accuracy:", grid_search.score(X_train_tfidf, y_train))
print("Validation accuracy:", grid_search.score(X_val_tfidf, y_val))
print("Test accuracy:", grid_search.score(X_test_tfidf, y_test))
print("Best parameters:", grid_search.best_params_)

