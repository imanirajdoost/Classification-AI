# see this link for more info : https://towardsdatascience.com/5-simple-ways-to-tokenize-text-in-python-92c6804edfc4
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# @TODO: IN PROGRESS : The goal is to combine tokenize_text.py and app.py and to use both digits and text to predict new data

pd.options.mode.chained_assignment = None  # default='warn'
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# nltk.download('stopwords')
# nltk.download('punkt')


# Define a function to pre-process each text
def preprocess_text(all_text):
    temp = all_text
    final_tokens = all_text
    # Remove stop words
    stop_words = set(stopwords.words("french"))
    stemmer = SnowballStemmer("french")
    # pre-process the data
    for index, value in enumerate(all_text):
        all_text[index] = value.lower().replace(',', '').replace('.', '').replace('?', '')
        temp[index] = word_tokenize(all_text[index], language='french')

    for j, line in enumerate(temp):
        filtered_tokens = [token for token in line if token not in stop_words]
        # Stem each token
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        # Join the stemmed tokens into a string
        final_tokens[j] = " ".join(stemmed_tokens)
    return final_tokens


data_digit = pd.read_csv('../binary_classification/digit_en_pred2.csv', sep=',', header=1)
data_digit_WithHeader = pd.read_csv('../binary_classification/digit_en_pred2.csv', sep=',')

should_scale_data = True

y = data_digit.iloc[:, data_digit.shape[1] - 1]
X_digit = data_digit.iloc[:, :data_digit.shape[1] - 1]

# X_train_digit, X_test_digit, y_train_digit, y_test_digit = train_test_split(X.values, y.values, test_size=0.2)

print('=======================')

# preprocessing data
if should_scale_data:
    scaled_data_X = preprocessing.StandardScaler().fit_transform(X_digit)
else:
    scaled_data_X = X_digit

data = pd.read_csv('../tokenize_text/text_fr_learn.csv', sep=',', header=1)
dataWithHeader = pd.read_csv('../tokenize_text/text_fr_learn.csv', sep=',')

# get the first column as text data column
text_data = data.iloc[:, 0]

# get the last column as the result target classifcation
target = data.iloc[:, data.shape[1] - 1]

# print(dataWithHeader.describe())

result = preprocess_text(text_data)
# result = pd.DataFrame(result)

# initialize
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

# Fit and transform the text data to create the bag-of-words representation
X = vectorizer.fit_transform(result)

# create document term matrix

scaled_data_X = np.array(scaled_data_X)
X = np.array(X.toarray())

con = np.concatenate((X, scaled_data_X), axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(con, target, test_size=0.2)

# Neural Network
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
NN.fit(con, y_train)
NN.predict(X_test)

scores_nn = cross_val_score(NN, X_test, y_test, cv=10)
print('Accuracy of NN: ', scores_nn)
