#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: romanshen 
@file: preprocess_eng.py 
@time: 2021/03/05
@contact: xiangqing.shen@njust.edu.cn
"""

import re

from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords as sw, wordnet as wn
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import string


def preprocess(document, max_features=150, max_sentence_len=300):
    """
    Returns a normalized, lemmatized list of tokens from a document by
    applying segmentation (breaking into sentences), then word/punctuation
    tokenization, and finally part of speech tagging. It uses the part of
    speech tags to look up the lemma in WordNet, and returns the lowercase
    version of all the words, removing stopwords and punctuation.
    """

    def lemmatize(token, tag):
        """
        Converts the tag to a WordNet POS tag, then uses that
        tag to perform an accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return WordNetLemmatizer().lemmatize(token, tag)

    def vectorize(doc, max_features, max_sentence_len):
        """
        Converts a document into a sequence of indices of length max_sentence_len retaining only max_features unique words
        """
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(doc)
        doc = tokenizer.texts_to_sequences(doc)
        doc_pad = pad_sequences(doc, padding='pre', truncating='pre', maxlen=max_sentence_len)
        return np.squeeze(doc_pad), tokenizer.word_index

    cleaned_document = []
    vocab = []

    # Break the document into sentences
    for sent in document:

        # Clean the text using a few regular expressions
        sent = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", sent)
        sent = re.sub(r"what's", "what is ", sent)
        sent = re.sub(r"\'", " ", sent)
        sent = re.sub(r"@", " ", sent)
        sent = re.sub(r"\'ve", " have ", sent)
        sent = re.sub(r"can't", "cannot ", sent)
        sent = re.sub(r"n't", " not ", sent)
        sent = re.sub(r"i'm", "i am ", sent)
        sent = re.sub(r"\'re", " are ", sent)
        sent = re.sub(r"\'d", " would ", sent)
        sent = re.sub(r"\'ll", " will ", sent)
        sent = re.sub(r"(\d+)(k)", r"\g<1>000", sent)
        sent = sent.replace("\n", " ")

        lemmatized_tokens = []

        # Break the sentence into part of speech tagged tokens
        for token, tag in pos_tag(wordpunct_tokenize(sent)):

            # Apply preprocessing to the tokens
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # If punctuation ignore token and continue
            if all(char in set(string.punctuation) for char in token) or token in set(sw.words('english')):
                continue

            # Lemmatize the token
            lemma = lemmatize(token, tag)
            lemmatized_tokens.append(lemma)
            vocab.append(lemma)

        cleaned_document.append(lemmatized_tokens)

    vocab = sorted(list(set(vocab)))

    return cleaned_document, vocab


def build_vocab(file_path, count_thr=2):
    with open(file_path, 'r', encoding='utf-8') as f:
        counts = {}
        for line in f:
            line = line.strip()
            words = line.split(' ')
            for word in words:
                counts[word] = counts.get(word, 0) + 1
    vocab = [w for w, n in counts.items() if n >= count_thr]
    return vocab


def generate_bow(file_path, ignore_freq=True):
    # year not in vocab,
    vocab = ["volleyball", "beijing", "china", "institute", "win", "go", "champion", "computer", "science",
             "technology", "human", "race", "university", "artificial", "intelligence", "machine", "game",
             "competition", "spring", "young", "green", "grass"]
    bows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            bow = np.zeros(len(vocab))
            line = line.strip()
            words = line.split(' ')
            if ignore_freq:
                for i, word in enumerate(vocab):
                    if word in words:
                        bow[i] = 1
            else:
                for word in words:
                    for i, w in enumerate(vocab):
                        if w == word:
                            bow[i] += 1
            bows.append(bow)
    return np.vstack(bows)


if __name__ == '__main__':
    docs = []
    # nltk.download('averaged_perceptron_tagger')
    with open('./eng_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            docs.append(line)
    d, vocab = preprocess(docs)

    with open("./real_clean_eng_data.txt", "w", encoding="utf-8") as f:
        for s in d:
            f.write(" ".join(s))
            f.write("\n")

    # with open("./clean_eng_data.txt", "r", encoding="utf-8") as f:
    #     filtered = []
    #     vocab = ["volleyball", "beijing", "china", "institute", "win", "go", "champion", "computer", "science",
    #              "technology", "human", "race", "university", "artificial", "intelligence", "machine", "game",
    #              "competition", "spring", "young", "green", "grass"]
    #     for line in f:
    #             line = line.strip()
    #             words = line.split(' ')
    #             tmp = []
    #             for word in words:
    #                 if word in vocab:
    #                     tmp.append(word)
    #             filtered.append(tmp)
    #     with open("./vec_eng_data.txt", "w", encoding="utf-8") as fi:
    #         for s in filtered:
    #             fi.write(" ".join(s))
    #             fi.write("\n")

    # generate_bow('./clean_eng_data.txt')
