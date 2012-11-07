import copy
import numpy
import random
import sys
import math
from consts import *
from util import find_max_density

from nltk.cluster.util import VectorSpaceClusterer
from tfidf import TF_IDF

def gaac_demo():
    """
    Non-interactive demonstration of the clusterers with simple 2-D data.
    """

    from nltk.cluster import GAAClusterer

    # use a set of tokens with 2D indices
    vectors = [numpy.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

    # test the GAAC clusterer with 4 clusters
    clusterer = GAAClusterer(4)
    clusters = clusterer.cluster(vectors, True)

    print 'Clusterer:', clusterer
    print 'Clustered:', vectors
    print 'As:', clusters
    print

    # show the dendrogram
    clusterer.dendrogram().show()

    # classify a new vector
    vector = numpy.array([3, 3])
    print 'classify(%s):' % vector,
    print clusterer.classify(vector)
    print

def kmeans_demo():
    # example from figure 14.9, page 517, Manning and Schutze

    from nltk.cluster import KMeansClusterer, euclidean_distance

    vectors = [numpy.array(f) for f in [[2, 1], [1, 3], [4, 7], [6, 7]]]
    print 'numpy:', vectors
    means = [[4, 3], [5, 5]]

    clusterer = KMeansClusterer(2, euclidean_distance, initial_means=means)
    clusters = clusterer.cluster(vectors, True, trace=True)

    print 'Clustered:', vectors
    print 'As:', clusters
    print 'Means:', clusterer.means()
    print

    vectors = [numpy.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

    # test k-means using the euclidean distance metric, 2 means and repeat
    # clustering 10 times with random seeds

    clusterer = KMeansClusterer(2, euclidean_distance, repeats=10)
    clusters = clusterer.cluster(vectors, True)
    print 'Clustered:', vectors
    print 'As:', clusters
    print 'Means:', clusterer.means()
    print

    # classify a new vector
    vector = numpy.array([3, 3])
    print 'classify(%s):' % vector,
    print clusterer.classify(vector)

def my_demo_main(file_list_name, tokenizer_num=0):
    from mmseg import seg_txt
    from nltk.cluster import KMeansClusterer, euclidean_distance
    from nltk.cluster import GAAClusterer
    tokenizer_list = [seg_txt,]
    file_list = open(file_list_name)
    tokenizer = tokenizer_list[tokenizer_num]
    texts = [[term for term in tokenizer(open('pos/' + str(file_name.strip())).read())] for file_name in file_list]

    data = TF_IDF(texts)

    vectors = []

    file_count = 1
    feature_set = set()
    for text in data.texts:
        vector = list()
        for term in set(text):
            vector.append((data.tf_idf(term, text), term))
        vector.sort(key=lambda x:x[0], reverse=True)
        for term in vector[:int(len(vector)*0.15) + 1]:
            feature_set.add(term[1])

    print feature_set
    print len(feature_set)
    for text in data.texts:
        vector = list()
        for term in feature_set:
            if term in text:
                vector.append(data.tf_idf(term, text))
            else:
                vector.append(0)
        square_sum = map(lambda x:x*x, vector)
        square_sum = math.sqrt(sum(square_sum))
        vector = map(lambda x:x/square_sum, vector)
        vectors += [numpy.array(vector)]
        print file_count
        file_count += 1

    means = find_max_density(vectors, euclidean_distance);
    print 'means', len(means)

    f = open('result.txt', 'w')
    clusterer = KMeansClusterer(len(means), euclidean_distance, initial_means=means)
    clusters = clusterer.cluster(vectors, True, True)
    print 'km1', clusters
    f.write('km1: ' + str(clusters) + '\n')

    clusterer = KMeansClusterer(len(vectors) / 10, euclidean_distance, repeats=10)

    clusters = clusterer.cluster(vectors, True, True)
    print 'km2', clusters
    f.write('km2: ' + str(clusters) + '\n')

    clusterer = GAAClusterer(len(vectors) / 10)
    clusters = clusterer.cluster(vectors, True)
    print 'gaac', clusters
    f.write('gaac: ' + str(clusters) + '\n')
    f.close()

if __name__ == '__main__':
    if sys.argv[1] == 'gaac':
        gaac_demo()
    elif sys.argv[1] == 'km':
        kmeans_demo()
    elif sys.argv[1] == 'demo':
        my_demo_main(sys.argv[2])

