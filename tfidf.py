# -*- coding: utf-8 -*-

import nltk
from math import log

class TF_IDF():
    def __init__(self, texts):
        print 'hello'
        self.texts = texts

        self._texts_num = len(self.texts)
        self._terms_index = self._index_and_cal_terms_num()

        self.terms_set = self._terms_index.keys()
        self._terms_num = len(self.terms_set)
        print 'terms_num ', self._terms_num
        self._terms_match_texts = self._cal_terms_match_texts()

    def _index_and_cal_terms_num(self):
        index = 0
        terms_index = dict()
        for text in self.texts:
            for term in text:
                if terms_index.get(term) is None:
                    terms_index[term] = index
                    print index, ' ', term
                    index += 1
        print 'index ', index
        print len(terms_index)
        return terms_index

    def _cal_terms_match_texts(self):
        terms_match_texts = [0 for i in range(self._terms_num)]

        for text in self.texts:
            has_match_text = [False for i in range(self._terms_num)]
            for term in text:
                if self._terms_index.get(term) is None:
                    print 'term_error', term
                    return
                term_index = self._terms_index[term]
                if term_index > len(has_match_text):
                    print 'term_index ', term_index
                    return 
                if not has_match_text[term_index]:
                    has_match_text[term_index] = True
                    terms_match_texts[term_index] += 1
        
        return [log(self._terms_num/float(term)) for term in terms_match_texts]

    def tf(self, term, text):
        return float(text.count(term)) / len(text)

    def idf(self, term):
        if self._terms_index.get(term) is None:
            print 'error in idf\'s term'
            return 

        return self._terms_match_texts[self._terms_index[term]]

    def tf_idf(self, term, text):
        return self.tf(term, text) * self.idf(term)


