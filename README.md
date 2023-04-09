# IA de la traduction

## IA de la classification

L'IA de la classification prend en compte les datas de la traduction de [ce jeu de donnée](https://docs.google.com/spreadsheets/d/1_i6kVy7n9s72CsCWyN6i1Zqp8jxh_uYjvKRbLvIIY2k/edit#gid=1440583160).

### Description

L'IA du fichier `binary_classification_sklearn.py` est entraînée avec cette data `digit_en_learnX.csv` et ensuite sera évaluée par les datas du fichier `digit_en_predX.csv`.

Pour tester les autres jeux de données il vous suffit de changer les contenus de ces fichiers ou utiliser les fichiers existants qui terminent par `_learn`. **Attention de garder le même format pour de nouveaux datas.**

Chaque dataset est différent et donnera un résultat différent.

Les fichiers qui commencent par `digit` sont les données numériques. Les fichiers qui commencent par `text` sont les données textes.

Les fichiers qui commencent par `ml_review_` sont les données sans classement (Pas de validation). Ces datas sont utilisées
 pour la prédiction sans la validation.

### Data d'entraînement

Dans le fichier `digit_X.csv` les colonnes sont :
- distance (La distance Levenshtein)
- bleu (Le score BLEU)
- ter
- tri
- lcs
- includesFormula (0 ou 1)
- onlyNumbers     (0 ou 1)
- includesBraces  (0 ou 1)
- hasFormatting   (0 ou 1)
- numberOfWords
- includesLink    (0 ou 1)
- reviewScore     (entre 0 et 1 ; il est calculé en fonction des autres éléments et leurs importances)
- AI_trad_is_good (0 ou 1 ; le résultat choisi)


### Data prediction

La même data comme la data d'entraînement sauf la dernière colonne `AI_trad_is_good` qui sera prédit par l'IA.

## IA de la traduction

Le fichier `machine_translation.py`. Cette IA est entraînée avec les données textes du fichier `fra.txt`, ce jeu de donnée est téléchargé depuis [ce lien](http://www.manythings.org/anki/). Cette IA est limitée pour les textes du projet et ne donne pas le bon résultat mais, pourra être intéressant dans l'avenir pour des textes plus ou moins avancé ; par exemple en faisant un glossaire Lean.

## IA de classification avec tokenisation

`tokenize_text.py` : Cette IA utilise le même jeu de donnée de classification mais, malgré l'autre, celle-ci utilise les textes 'tokenisés'. Ceci fait l'apprentissage avec le texte français. Pour plus d'info de comment `CountVectorizer tokenizer` fonctionne, [voir ici](https://kavita-ganesan.com/how-to-use-countvectorizer/#.Y-ZvBHbMJD8).

Les fichiers `text_fr_learnX.csv` et `text_fr_predictionX.csv` sont les datas d'apprentissage et les datas tests de cette IA.

## L'IA de Tokenize and classify [WIP]

`tokenize_text_and_classify.py` : Cette IA est une combinaison de l'IA de tokenization du texte et la classification binaire.
(`binary_classification_sklearn.py` et `tokenize_text`). Ceci est un WORK IN PROGRESS.

# Les outils automatiques

Les fichiers à mettre sur le serveur pour automatiser le process du machine learning et prédiction.

## Usage and installation

Install [Python 3+](https://www.python.org/downloads/) on your machine.

You can run the `.py` scripts using the command `python [FILE_NAME]`

install the Python libraries using `pip` command:


``
pip install pandas
``

``
pip install sklearn
``

``
pip install nltk
``

Execute this line to install the required packages in /python folder
``python -m pip install -r requirements.txt -t ./python``

## Auto ML Text

Ce fichier prend en entrée avec l'option `-i` un chemin vers un fichier JSON qui contient les textes sur lesquels on
voudra faire la prédiction. Le résultat sera retourné en JSON. Le résultat de ce programme est 0 ou 1. 0 veut dire que
le texte original sera potentiellement mieux que le texte traduit par Deepl et l'inverse.

Pour voir les options de ce fichier, exécutez le script avec l'option `-h` ou `--help`

## Auto ML Number

Ce fichier prend en entrée avec l'option `-i` un chemin vers un fichier JSON qui contient les chiffres (review score, 
distance, bleu, ter, tri, lcs, numberOfWords) sur lesquels on voudra faire la prédiction.

Ce fichier se ressemble au fichier `binary_classification_sklearn.py`.
Les régles sont les mêmes que pour ce fichier. Le résultat sera retourné en JSON. Le résultat de ce programme est 0 ou 1.
0 veut dire que le texte original sera potentiellement mieux que le texte traduit par Deepl et l'inverse.

Pour voir les options de ce fichier, exécutez le script avec l'option `-h` ou `--help`