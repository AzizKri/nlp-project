from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK files
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download("maxent_ne_chunker")
# nltk.download("words")
# nltk.download('twitter_samples')
# nltk.download('vader_lexicon')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Set stopwords & stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()



tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]
from random import shuffle

def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0

shuffle(tweets)
for tweet in tweets[:10]:
    print(">", is_positive(tweet), tweet)


#
# # Chunking and chinking
# grammar1 = "NP: {<DT>?<JJ>*<NN>}"
# grammar2 = """
# Chunk: {<.*>+}
# }<JJ>{"""
# chunk_parser = nltk.RegexpParser(grammar2)
#
# example_string = """
# Muad'Dib learned rapidly because his first training was in how to learn.
# And the first lesson of all was the basic trust that he could learn.
# It's shocking to find how many people do not believe they can learn,
# and how many more believe learning to be difficult."""
#
# # Tokenize
# words = word_tokenize(example_string)
#
# # Find stems (roots) of each word in the list
# stemmed_words = [stemmer.stem(word) for word in words]
#
# # Tag words
# tagged_words = nltk.pos_tag(words)
#
# # Chunk the phrases
# # tree = chunk_parser.parse(tagged_words)
# tree = nltk.ne_chunk(tagged_words)
# tree.draw()
#
# # Lemmatize the words
# lemmatized_words = []
# for word, tag in tagged_words:
#     lemmatized_words.append(lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)))
#
# # Remove stopwords
# filtered_list = [
#     word for word in lemmatized_words if word.casefold() not in stop_words
# ]
#
#
# print(filtered_list)