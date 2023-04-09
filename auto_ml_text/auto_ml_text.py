# see this link for more info : https://towardsdatascience.com/5-simple-ways-to-tokenize-text-in-python-92c6804edfc4
import os
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

pd.options.mode.chained_assignment = None  # default='warn'
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# nltk.download('stopwords')
# nltk.download('punkt')

import time

app_version = '2.0.0'
should_train_model = False
should_log = False
model_to_use = 0
main_data = pd.DataFrame()

# Determine app execution time
start = time.process_time()

# Change this to french, etc.
source_language = 'english'

# List of all available commands
command_list = {"-v": "Shows tool's version", "--version": "Shows tool's version",
                "-h": "Show tool's help", "--help": "Show tool's help",
                "-t": "train the model with the local train.csv file",
                "--train": "train the model with the local train.csv file",
                "-m [NUM]": "Model type to use (0: MultinomialNB [default], 1: Neural Network)",
                "--model [NUM]": "Model type to use (0: MultinomialNB [default], 1: Neural Network)",
                "-i [PATH]": "path to input data to predict as JSON",
                "--input [PATH]": "path to input data to predict as JSON",
                "-l": "log progress on screen", "--log": "log progress on screen"}

# Check for help command
for i in range(0, len(sys.argv)):
    if sys.argv[i] == '-h' or sys.argv[i] == '--help':
        print('This tool will train a sklearn Machine Learning model and uses text tokenization to predict new data.')
        print('The input option is mandatory (-i or --input)')
        print('If no training model exist for the given model, model will be trained and saved to file\n')
        j = 0
        while j < len(command_list) - 1:
            print(list(command_list)[j], ",", list(command_list)[j + 1], ":", list(command_list.values())[j])
            j += 2
        exit(0)

# Check for version command
for i in range(0, len(sys.argv)):
    if sys.argv[i] == '-v' or sys.argv[i] == '--version':
        print(f'version: {app_version}')
        exit(0)

# Check for train command
for i in range(0, len(sys.argv)):
    if sys.argv[i] == '-t' or sys.argv[i] == '--train':
        should_train_model = True
        break

# Check for log command
for i in range(0, len(sys.argv)):
    if sys.argv[i] == '-l' or sys.argv[i] == '--log':
        should_log = True
        break

# Check for model command
for i in range(0, len(sys.argv)):
    if sys.argv[i] == '-m' or sys.argv[i] == '--model':
        # Make sure user has given a number for model type
        if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            # if the given number is not between the supported models, revert to default (0)
            if int(sys.argv[i + 1]) >= 0 or int(sys.argv[i + 1]) <= 1:
                model_to_use = int(sys.argv[i + 1])
            else:
                model_to_use = 0
            break
        else:
            exit("error while parsing model")

model_name = "MultinomialNB"
if model_to_use == 1:
    model_name = "Neural Network"

# Check for input command (must exist)
input_exists = False
for i in range(0, len(sys.argv)):
    if sys.argv[i] == '-i' or sys.argv[i] == '--input':
        if i + 1 < len(sys.argv):
            input_data = sys.argv[i + 1]
            # main_data = pd.read_csv('ml_test_data_en.csv')
            main_data = pd.read_json(input_data)
            # try:
            #     main_data = pd.read_json(input_data)
            # except:
            #     print("Error while parsing input JSON")
            #     exit("JSON parse error")
        else:
            exit("error while parsing input")
        input_exists = True

if not input_exists:
    exit("No input was given. use -i or --input to pass an input")


# function to log data
def log(text):
    if should_log:
        print(text)


# function to pre-process each text
def preprocess_text(all_text):
    temp = []
    final_tokens = []
    # Remove stop words
    stop_words = set(stopwords.words(source_language))
    stemmer = SnowballStemmer(source_language)
    # pre-process the data
    for index, value in enumerate(all_text):
        string_value = str(value)
        all_text[index] = string_value.lower().replace(',', '').replace('.', '').replace('?', '')
        # temp[index] = word_tokenize(all_text[index], language='french')
        temp.append(word_tokenize(all_text[index], language=source_language))

    for k, line in enumerate(temp):
        filtered_tokens = [token for token in line if token not in stop_words]
        # Stem each token
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        # Join the stemmed tokens into a string
        final_tokens.append(" ".join(stemmed_tokens))
    return final_tokens


# check if model already exists
filename = model_name + '.sav'

# initialize
vectorizer = CountVectorizer(binary=True)
# vectorizer = TfidfVectorizer()

# Create an instance of the model class
if model_to_use == 0:
    clf = MultinomialNB()
else:
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(645, 2), random_state=1, max_iter=1000)

if should_train_model or not os.path.isfile(filename):
    log("Starting training model...")

    train_data = pd.read_csv('train.csv', sep=',')

    # get the first column as text data column
    text_data = train_data.iloc[:, 0]

    result = preprocess_text(text_data)

    # Fit and transform the text data to create the bag-of-words representation
    X = vectorizer.fit_transform(result)

    # get the last column as the result target classification
    target = train_data.iloc[:, train_data.shape[1] - 1]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2)

    # Fit the classifier on the training data
    clf.fit(X_train, y_train)

    # Save Multinomial NB model and vectorizer to a file
    with open(filename, 'wb') as fout:
        pickle.dump((vectorizer, clf), fout)
        log("Model saved to " + filename)

    # Evaluate the classifier on the test data
    scores = cross_val_score(clf, X_test, y_test, cv=10)
    log(f'Accuracy of {model_name}: {str(scores.mean())}')
else:
    log("Loading model from " + filename)
    with open(filename, 'rb') as f:
        vectorizer, clf = pickle.load(f)

log("Starting prediction...")
log(main_data.describe())
unseen_data = main_data.iloc[:, 0]
unseen_pred_text = preprocess_text(unseen_data)

unseen_X = vectorizer.transform(unseen_pred_text)

# Predict the class of the new text data using the trained classifier
predictions_unseen = clf.predict(unseen_X)

predictions_df = pd.DataFrame(predictions_unseen)
log(predictions_df.describe())
# return result as JSON
print(predictions_df.to_json(orient='records'))
# print('done')

end = time.process_time()
log(f'app run time: {(end - start) * 1000} ms')
