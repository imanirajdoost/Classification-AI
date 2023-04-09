# see this link for more info : https://towardsdatascience.com/5-simple-ways-to-tokenize-text-in-python-92c6804edfc4
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

pd.options.mode.chained_assignment = None  # default='warn'
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# nltk.download('stopwords')
# nltk.download('punkt')

import bcolors

# Change this to french, english, etc.
source_language = 'french'

write_unseen_data = True


# Define a function to pre-process each text
def preprocess_text(all_text):
    temp = []
    final_tokens = []
    # Remove stop words
    stop_words = set(stopwords.words(source_language))
    stemmer = SnowballStemmer(source_language)
    # pre-process the data
    for index, value in enumerate(all_text):
        all_text[index] = value.lower().replace(',', '').replace('.', '').replace('?', '')
        # temp[index] = word_tokenize(all_text[index], language='french')
        temp.append(word_tokenize(all_text[index], language=source_language))

    for j, line in enumerate(temp):
        filtered_tokens = [token for token in line if token not in stop_words]
        # Stem each token
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        # Join the stemmed tokens into a string
        final_tokens.append(" ".join(stemmed_tokens))
    return final_tokens


data = pd.read_csv('text_fr_learn.csv', sep=',')
dataWithHeader = pd.read_csv('text_fr_learn.csv', sep=',')

# get the first column as text data column
text_data = data.iloc[:, 0]

# get the last column as the result target classifcation
target = data.iloc[:, data.shape[1] - 1]

print(dataWithHeader.describe())

result = preprocess_text(text_data)
# result = pd.DataFrame(result)

# initialize
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

# Fit and transform the text data to create the bag-of-words representation
X = vectorizer.fit_transform(result)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2)

# Create an instance of the MultinomialNB class
clf = MultinomialNB()

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Evaluate the classifier on the test data
scores = cross_val_score(clf, X_test, y_test, cv=10)
# print('Accuracy:', clf.score(X_test, y_test))
print('Accuracy (MultinomialNB):', scores)

# Predict new data
new_data = pd.read_csv('text_fr_prediction.csv', sep=',')

pred_text = new_data.iloc[:, 0]
pred_class = new_data.iloc[:, new_data.shape[1] - 1]

pre_proc_pred_text = preprocess_text(pred_text)

# Transform the new text data using the fitted CountVectorizer
new_X = vectorizer.transform(pre_proc_pred_text)

print('====================')
print('Predictions:')

# Predict the class of the new text data using the trained classifier
predictions = clf.predict(new_X)

# Print the predictions
i = 0
for item in predictions:
    col = bcolors.bcolors.OKCYAN
    if item != pred_class[i]:
        col = bcolors.bcolors.WARNING
    print(col, i, '.Result:', item, ' Expected:', pred_class[i], bcolors.bcolors.ENDC)
    i += 1

print('====================')
# Neural Network
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(X_train, y_train)
NN.predict(X_test)

# Evaluate the classifier on the test data
scores = cross_val_score(NN, X_test, y_test, cv=10)
print('Accuracy (Neural Network):', scores)

print('====================')
print('Predictions:')

predictions = NN.predict(new_X)

# Print the predictions
i = 0
for item in predictions:
    col = bcolors.bcolors.OKCYAN
    if item != pred_class[i]:
        col = bcolors.bcolors.WARNING
    print(col, i, '.Result:', item, ' Expected:', pred_class[i], bcolors.bcolors.ENDC)
    i += 1

if write_unseen_data:
    # Unseen data
    unseen_data_file = 'ml_review_text_fr.csv'
    unseen_data = pd.read_csv(unseen_data_file)
    print(unseen_data.describe())
    unseen_data = unseen_data.iloc[:, 0]
    unseen_pred_text = preprocess_text(unseen_data)

    unseen_X = vectorizer.transform(unseen_pred_text)

    print('====================')
    print('Predictions:')

    # Predict the class of the new text data using the trained classifier
    predictions_unseen = clf.predict(unseen_X)

    # Print the predictions
    i = 0
    result_one = 0
    result_zero = 0
    for item in predictions_unseen:
        col = bcolors.bcolors.OKCYAN
        if item == 1:
            col = bcolors.bcolors.WARNING
            result_one += 1
        else:
            result_zero += 1
        print(col, i, '.Result:', item, bcolors.bcolors.ENDC)
        i += 1

    predictions_df = pd.DataFrame(predictions_unseen)
    print(predictions_df.describe())
    print('====================')
    print(f'number of zeros: {result_zero}, number of ones: {result_one}')
    predictions_df.to_csv("result_" + unseen_data_file, header=False, index=False)
