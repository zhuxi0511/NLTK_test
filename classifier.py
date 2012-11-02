import random
import nltk
from nltk.corpus import names
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

word_features = tuple(all_words.keys()[:2000])

def document_features(document, word_features):
    document_words = set(document) 
    document_freqs = nltk.FreqDist(w.lower() for w in document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words) 
        #features['conut(%s)' % word] = 1 if document_freqs[word] >= 10 else 2

    return features

#print document_features(movie_reviews.words(movie_reviews.fileids()[0]), word_features)

featuresets = [(document_features(word, word_features), category) for word, category in documents]

train_set, dev_set, test_set = featuresets[:100], featuresets[100:200], featuresets[200:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print classifier

print nltk.classify.accuracy(classifier, dev_set)

classifier.show_most_informative_features(5)
