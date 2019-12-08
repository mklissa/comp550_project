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
import numpy as np
from scipy.special import lambertw
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
        print(len(sequence))
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
    
    alpha=0.5
    all_hidden_states = A.conditions()
    num_states = len(all_hidden_states)
    T=200
    p_qt = np.zeros((T,num_states))
    for i,l in enumerate(all_hidden_states):
        p_qt[0,i] = pi.prob(l)
    
    for t in range(1,T):
        for i,m in enumerate(all_hidden_states):
            for j,l in enumerate(all_hidden_states):
                p_qt[t,i] += A.get(l).prob(m) * p_qt[t-1,j]

    sum_t_p_qt = p_qt.sum(axis=0)
    W = np.zeros((num_states,len(symbols)))
    for q,i in enumerate(all_hidden_states): # Q loop
        for o,j in enumerate(symbols): # O loop
            if outputs.get(i).get(j):
                W[q,o] = outputs.get(i).get(j) * alpha / ((1-alpha) * sum_t_p_qt[q])
    gamma=-10
    g = np.zeros((num_states))
    for q,i in enumerate(all_hidden_states): # Q loop
        g[q] = gamma * alpha / ((1-alpha) * sum_t_p_qt[q])

    b_prob = np.zeros((num_states,len(symbols)))
    for i in range(len(all_hidden_states)): # Q loop
        for j in range(len(symbols)): # O loop
            if W[i,j] ==0.:
                
                b_prob[i,j] = 0.
            else:

                for gamma in np.linspace(-1000,1000,8000):
                    g= gamma * alpha / ((1-alpha) * sum_t_p_qt[i])
                    input_lamb= - W[i,j] * np.exp(1+ g)
                    # print(g) #-9 is needed
                    # print('input_lamb',  input_lamb)
                    # print('lambert value',lambertw(input_lamb ,-1))
                    # print('b prob', - W[i,j] / lambertw(input_lamb,-1 ))
                    b_candidate = - W[i,j] / lambertw(input_lamb,-1 )
                    if abs(b_candidate - 1.) <= 0.1:
                        b_prob[i,j] = b_candidate

    # for k in outputs.keys():
    #     outputs.get(k)
    import pdb;pdb.set_trace()
    print(b_prob[0])
    
    return hmm.HiddenMarkovModelTagger(symbols, states, A, B, pi)



