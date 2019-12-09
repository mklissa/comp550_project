import nltk
import numpy as np
from nltk.tag import hmm
import nltk.probability as pr
from train_sup import train_supervised
import re

import argparse

import random

np.random.seed(1)
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("-laplace",
                    action="store_true")
parser.add_argument("-mihmm",
                    action="store_true")
parser.add_argument("-lm",
                    action="store_true")
parser.add_argument("data_id")
parser.add_argument("-pos",
                    action="store_true")
args = parser.parse_args()

states = symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']

extra_sents = []
if args.lm:
    nltk.download('brown')
    from nltk.corpus import brown

    cats = ['news']

    for cat in cats:
        extra_data = brown.sents(categories=cat)
        for sent in extra_data:
            str_sent = ' '.join(sent)
            process = ''
            for i in range(len(str_sent)):
                if str_sent[i].lower() not in states:
                    continue
                if str_sent[i] == ' ' and not process:
                    continue
                if str_sent[i] == ' ' and i != len(str_sent) - 1 and (
                        str_sent[i + 1] == '.' or str_sent[i + 1] == ',' or process[-1] == ' '):
                    continue
                process += str_sent[i].lower()

            extra_sents.append(process)

with open('{}/train_cipher.txt'.format(args.data_id), 'r') as f:
    data = f.read()
cipher = data.split('\n')
if '' in cipher:
    cipher.remove('')

with open('{}/train_plain.txt'.format(args.data_id), 'r') as f:
    data = f.read()
plain = data.split('\n')
if '' in plain:
    plain.remove('')

train_data = []
for c_sent, p_sent in zip(cipher, plain):
    sent_tuples = [(c_sent[i], p_sent[i]) for i in range(len(c_sent))]
    train_data.append(sent_tuples)

if args.laplace:
    estim = lambda fd, bins: nltk.LaplaceProbDist(fd, bins)
else:
    estim = lambda fdist, bins: nltk.MLEProbDist(fdist)

# Train HMM on POS tagging instead of ciphers
test_data = []
if args.pos:
    # nltk.download('brown')
    # nltk.download('universal_tagset')
    from nltk.corpus import brown

    # list of (list of (str,str)), each top level list is a sentence, containing (word,tag) pairs
    brown_news_tagged = brown.tagged_sents(categories='news', tagset='universal')[:1000]
    n = len(brown_news_tagged)
    # import pdb;pdb.set_trace()
    # Clean up sentences from brown and build sets of states and symbols
    tag_re = re.compile(r'[*]|--|[^+*-]+')
    tag_set = set()
    symbols = set()

    del train_data[:]
    for sentence in brown_news_tagged[:int(n * 0.8)]:
        for i in range(len(sentence)):
            word, tag = sentence[i]
            word = word.lower()  # normalize
            symbols.add(word)  # log this word
            # Clean up the tag.
            tag = tag_re.match(tag).group()
            tag_set.add(tag)
            sentence[i] = (word, tag)  # store cleaned-up tagged token
        train_data += [sentence]

    # re-assign states and symbols for POS-tagging task
    states = list(tag_set)
    symbols = list(symbols)

    # Set up test data set by cleaning in the same manner
    for sentence in brown_news_tagged[int(n * 0.8):]:
        for i in range(len(sentence)):
            word, tag = sentence[i]
            word = word.lower()  # normalize
            # Clean up the tag.
            tag = tag_re.match(tag).group()
            sentence[i] = (word, tag)  # store cleaned-up tagged token
        test_data += [sentence]

tagger = train_supervised(states, symbols, train_data, estimator=estim, extra_sents=extra_sents,mihmm=args.mihmm)

# We redefine the test_data set if not doing POS tagging
if not args.pos:
    with open('{}/test_cipher.txt'.format(args.data_id), 'r') as f:
        data = f.read()
    test_cipher = data.split('\n')
    if '' in test_cipher:
        test_cipher.remove('')

    print("==================================")
    print("Generating test sentences for dataset {}".format(args.data_id))
    print("Laplace Smoothing is {}".format(args.laplace))
    print("Extra data is {}".format(args.lm))
    print("==================================")
    for test_sent in test_cipher:
        print('')
        tagged = tagger.best_path(test_sent)
        print(''.join(tagged))

    with open('{}/test_plain.txt'.format(args.data_id), 'r') as f:
        data = f.read()
    test_plain = data.split('\n')
    if '' in test_plain:
        test_plain.remove('')

    # test_data=[]
    for c_sent, p_sent in zip(test_cipher, test_plain):
        sent_tuples = [(c_sent[i], p_sent[i]) for i in range(len(c_sent))]
        test_data.append(sent_tuples)

print("\n==================================")
# train_acc = tagger.evaluate(train_data)
test_acc = tagger.evaluate(test_data)
# print("Training accuracy: {}".format(train_acc))
print("Test accuracy: {}".format(test_acc))
print("==================================")
