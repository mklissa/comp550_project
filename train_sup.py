from nltk.probability import (
    FreqDist,
    ConditionalFreqDist,
    ConditionalProbDist,
    DictionaryProbDist,
    DictionaryConditionalProbDist,
    LidstoneProbDist,
    MutableProbDist,
    MLEProbDist,
    RandomProbDist,
)
from nltk.tag import hmm
def train_supervised(states, symbols, labelled_sequences, estimator=None, extra_sents=None):

    # default to the MLE estimate
    if estimator is None:
        estimator = lambda fdist, bins: MLEProbDist(fdist)

    # count occurrences of starting states, transitions out of each state
    # and output symbols observed in each state
    known_symbols = set(symbols)
    known_states = set(states)


    # Create and define starting, transitions and outputs frequencies.
    starting = FreqDist()
    transitions = ConditionalFreqDist()
    outputs = ConditionalFreqDist()
    for sequence in labelled_sequences:
        lasts = None
        for token in sequence:
            state = token[1]
            symbol = token[0]
            if lasts is None:
                starting[state] += 1
            else:
                transitions[lasts][state] += 1
            outputs[state][symbol] += 1
            lasts = state


    # Add extra data for transition frequencies
    if extra_sents:
        mod_starting_freq = True # Change to True to also modify starting's frequency count
        for sequence in extra_sents:
            lasts = None
            for state in sequence:
                if lasts is None and mod_starting_freq:
                    starting[state] += 1
                else:
                    transitions[lasts][state] += 1                
                lasts = state

    # create probability distributions (with smoothing)
    N = len(states)
    pi = estimator(starting, N)
    A = ConditionalProbDist(transitions, estimator, N)
    B = ConditionalProbDist(outputs, estimator, len(symbols))

    return hmm.HiddenMarkovModelTagger(symbols, states, A, B, pi)