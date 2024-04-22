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
from sklearn.metrics import accuracy_score


# Read out dataset
data = pandas.read_csv('data2.csv', encoding='latin-1')
# data.values[0][0] -> Sentence
# data.values[0][1] -> Sentiment
# 0: Negative
# 2: Neutral
# 4: Positive


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
    if os.path.isfile('preprocessed_data.csv'):
        print("Loading previous data...")
        reader = csv.reader(open('preprocessed_data.csv'))
        previous_data = list(reader)
        preprocessed_data = [" ".join(row) for row in previous_data]
        return preprocessed_data
    else:
        print("Preprocessing...")
        doc = []
        doc_words = []
        for row in document.values:
            new_row_words = preprocess_single(row[0])
            doc_words.append(new_row_words)
            new_row = " ".join(new_row_words)
            doc.append(new_row)
        print("Preprocessed!")
        print("Saving...")
        save_to_csv(doc_words, "preprocessed_data.csv")
        print("Saved!")
        return doc


# Save data
def save_to_csv(pdata, filename):
    with open(filename, 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(pdata)


def train(vectors):
    # X is the sentence
    # Y is the label
    print("Labeling...")
    labels = []
    for row in data.values:
        labels.append(row[1])

    print("Splitting...")
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=82)

    print("Training...")
    # Initialize and train the model
    model = RandomForestClassifier(n_jobs=-1, verbose=1)
    model.fit(x_train, y_train)
    pickle.dump((tfidf_vectorizer, model), open("files/model.pkl", "wb"))

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
    tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed_data)

    # Train out data using the logistic regression method
    train(tfidf_vectors)


def main():
    build_model()
    vectorizer, model = pickle.load(open("files/model.pkl", "rb"))

    # text = input("Enter the text you would like to analyze: ")
    # while text != "0":
    #     preprocessed_text = preprocess_single(text)
    #     tfidf_vectors = vectorizer.transform(preprocessed_text)
    #     prediction = model.predict(tfidf_vectors)
    #     print(prediction)
    #     text = input("Enter the text you would like to analyze: ")

    for i in range(10):
        text = data.values[random.randint(0, 2999)][0]
        preprocessed_text = preprocess_single(text)
        tfidf_vectors = vectorizer.transform(preprocessed_text)
        prediction = model.predict(tfidf_vectors)
        print(text)
        print(preprocessed_text)
        print(prediction)
        print("---------------")


main()
