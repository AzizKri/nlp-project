import ast
import csv
import os
import random
import pandas
import pickle
from spellchecker import SpellChecker
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score

# Read out dataset
# data.values[0][0] -> Sentiment
# data.values[0][1] -> Sentence
# 0: Negative
# 4: Positive
data = pandas.read_csv('files/data2.csv', encoding='latin-1')

# Initialize
stemmer = SnowballStemmer('english')
tk = TweetTokenizer()
sc = SpellChecker()
stop_words = set(stopwords.words('english'))
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # Maximum number of features (words) to consider
    min_df=5,           # Ignore terms that have a document frequency strictly lower than the given threshold
    max_df=0.7,         # Ignore terms that have a document frequency strictly higher than the given threshold
    stop_words='english'  # Remove English stopwords
)


# Preprocessing sentences
def preprocess_single(tweet):
    # Here we chose to tokenize words then correct them as social media comments are full of misspellings and such.
    # We make sure to ignore @mentions and non-alphabetical letters
    words = tk.tokenize(tweet)
    corrected_words = []

    for word in words:
        if word.isalpha():
            corrected_words.append(word)

    # # Currently commented out because it takes extremely long and we were tight on time. Comment the above for loop
    # # and uncomment the below if you would like to correct misspellings

    # for word in words:
    #     if word.isalpha() and word[0].isupper():
    #         corrected = sc.correction(word)
    #         corrected_words.append(corrected or word)
    #     else:
    #         corrected_words.append(word)

    # We decided to stem words instead of lemmatizing them as it's faster and lemmatizing is not necessary
    stemmed_words = [stemmer.stem(word) for word in corrected_words]

    # Finally we remove the stop words as they often give no value to the sentence in this method
    preprocessed_words = [word for word in stemmed_words if word.casefold() not in stop_words]
    return preprocessed_words


# Preprocessing an entire pandas dataset
def preprocess_document(document):
    print("Checking for previous preprocessed data...")
    if os.path.isfile('files/preprocessed_data.csv'):
        print("Loading previous data...")
        reader = csv.reader(open('files/preprocessed_data.csv'))
        previous_data = list(reader)
        preprocessed_data = [(row[0], ast.literal_eval(row[1])) for row in previous_data]
        return preprocessed_data
    else:
        print("Preprocessing...")
        doc = []
        for row in document.values:
            new_row = preprocess_single(row[1])
            if len(new_row) < 1:
                continue
            else:
                doc.append((row[0], new_row))
        print("Preprocessed! Saving...")
        save_to_csv(doc, "files/preprocessed_data.csv")
        print("Saved!")
        return doc


# Save data
def save_to_csv(pdata, filename):
    with open(filename, 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(pdata)


def train(vectors, labels):
    # X is the sentence
    # Y is the label

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=82)

    # Initialize and train the model
    # Try RandomForestClassifier, LogisticRegression, BernoulliNB, MultinomialNB
    model = RandomForestClassifier(n_jobs=7, verbose=2, max_depth=128, n_estimators=100)
    # n_jobs is for how many CPU threads it should use
    # verbose is the level of logging that it should print while training
    # max_depth is for the maximum depth within a tree
    # n_estimators is the amount of decision trees in the forest
    print("Initialized model. Training...")
    model.fit(x_train, y_train)
    print("Dumping...")
    pickle.dump((tfidf_vectorizer, model), open("files/model.pkl", "wb"))

    print("Predicting")
    # Predict sentiment labels on the testing set
    y_pred = model.predict(x_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def build_model():
    # Start by preprocessing the data
    preprocessed_data = preprocess_document(data)

    # Fit the vectorizer to the preprocessed data and transform the data into TF-IDF vectors
    # This gives us, for every row of data, the TF-IDF score for every word in every sentence
    # (Sentence, Vocab) TF-IDF Score
    vocabs = [" ".join(row[1]) for row in preprocessed_data]
    sentiment = [row[0] for row in preprocessed_data]
    tfidf_vectors = tfidf_vectorizer.fit_transform(vocabs)

    # Train out data
    train(tfidf_vectors, sentiment)


def main():
    build_model()
    vectorizer, model = pickle.load(open("files/model.pkl", "rb"))

    text_mode = 0
    # 0 Custom
    # 1 Random from dataset
    if text_mode == 0:
        # Custom text
        text = input("Enter the text you would like to analyze: ")
        while text != "0":
            preprocessed_text = preprocess_single(text)
            if len(preprocessed_text) < 1:
                print("Sentence too short")
            else:
                tfidf_vectors = vectorizer.transform(preprocessed_text)
                prediction = model.predict(tfidf_vectors)
                print(prediction)
                pos = 0
                for num in prediction:
                    if num == 4:
                        pos += 1
                result = (pos/len(prediction))
                print("Positive" if result >= 1/7 else "Negative")
            text = input("Enter the text you would like to analyze: ")
    elif text_mode == 1:
        # Random text from the dataset
        text = ""
        while text != "0":
            text = data.values[random.randint(0, 1000000)][1]
            print(text)
            preprocessed_text = preprocess_single(text)
            while len(preprocessed_text) < 1:
                text = data.values[random.randint(0, 1000000)][1]
                preprocessed_text = preprocess_single(text)
            tfidf_vectors = vectorizer.transform(preprocessed_text)
            prediction = model.predict(tfidf_vectors)
            print(prediction)
            pos = 0
            for num in prediction:
                if num == 4:
                    pos += 1
            result = (pos/len(prediction))
            print("Positive" if result >= 1/7 else "Negative")
            text = input("---------------")


main()
