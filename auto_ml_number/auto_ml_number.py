import os
import pickle
import sys
import time

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

app_version = '2.0.0'
should_train_model = False
should_log = False
model_to_use = 0
main_data = pd.DataFrame()

# should_plot = False
should_scale_data = True

# Determine app execution time
start = time.process_time()

# List of all available commands
command_list = {"-v": "Shows tool's version", "--version": "Shows tool's version",
                "-h": "Show tool's help", "--help": "Show tool's help",
                "-t": "train the model with the local train.csv file",
                "--train": "train the model with the local train.csv file",
                "-m [NUM]": "Model type to use (0: RandomForestClassifier [default], 1: Neural Network, 2: KNeighborsClassifier)",
                "--model [NUM]": "Model type to use (0: MultinomialNB [default], 1: Neural Network, 2: KNeighborsClassifier)",
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

model_name = "RandomForestClassifier"
if model_to_use == 1:
    model_name = "Neural Network"
elif model_to_use == 2:
    model_name = "KNN"

# Check for input command (must exist)
input_exists = False
for i in range(0, len(sys.argv)):
    if sys.argv[i] == '-i' or sys.argv[i] == '--input':
        if i + 1 < len(sys.argv):
            input_data = sys.argv[i + 1]
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


# check if model already exists
filename = model_name + '.sav'

# Create an instance of the model class
if model_to_use == 0:
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
elif model_to_use == 1:
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
else:
    clf = KNeighborsClassifier(n_neighbors=3)

if should_train_model or not os.path.isfile(filename):
    log("Starting training model...")

    train_data = pd.read_csv('train.csv', sep=',')

    y = train_data.iloc[:, train_data.shape[1] - 1]
    X = train_data.iloc[:, :train_data.shape[1] - 1]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)

    # get the last column as the result target classification
    target = train_data.iloc[:, train_data.shape[1] - 1]

    # preprocessing data
    if should_scale_data:
        scaled_data_X_train = preprocessing.StandardScaler().fit_transform(X_train)
        scaled_data_X_test = preprocessing.StandardScaler().fit_transform(X_test)
    else:
        scaled_data_X_train = X_train
        scaled_data_X_test = X_test

    # if should_plot:
    #     plt.plot(scaled_data_X_train)
    #     plt.show()

    # Fit the classifier on the training data
    clf.fit(scaled_data_X_train, y_train)

    # Save Multinomial NB model and vectorizer to a file
    with open(filename, 'wb') as fout:
        pickle.dump(clf, fout)
        log("Model saved to " + filename)

    # Evaluate the classifier on the test data
    scores = cross_val_score(clf, X_test, y_test, cv=10)
    log(f'Accuracy of {model_name}: {str(scores.mean())}')
else:
    log("Loading model from " + filename)
    with open(filename, 'rb') as f:
        clf = pickle.load(f)

log("Starting prediction...")
log(main_data.describe())
pred_data = np.array(main_data)

# data scaling
if should_scale_data:
    pred_data = np.array(pred_data)
    temp = np.zeros(shape=(pred_data.shape[0], pred_data.shape[1] - 1))
    temp[:, :] = pred_data[:, :-1]

    temp = preprocessing.StandardScaler().fit_transform(temp)

    temp2 = np.zeros(shape=(pred_data.shape[0], pred_data.shape[1]))
    temp2[:, :-1] = temp[:, :]
    temp2 = np.array(temp2)
    temp2[:, -1] = pred_data[:, -1]

    pred_data = temp2
# end of data scaling

# predicting data
# Predict the class of the new text data using the trained classifier
predictions_unseen = clf.predict(pred_data)

predictions_df = pd.DataFrame(predictions_unseen)
log(predictions_df.describe())
# return result as JSON
print(predictions_df.to_json(orient='records'))
# print('done')

end = time.process_time()
log(f'app run time: {(end - start) * 1000} ms')
