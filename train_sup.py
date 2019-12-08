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
        # print(len(sequence))
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
    T=100
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
    # gamma=-10
    # g = np.zeros((num_states))
    # for q,i in enumerate(all_hidden_states): # Q loop
    #     g[q] = gamma * alpha / ((1-alpha) * sum_t_p_qt[q])

    # import pdb;pdb.set_trace()

    # b_prob = np.zeros((num_states,len(symbols)))
    # for i in range(len(all_hidden_states)): # Q loop
    #     for j in range(len(symbols)): # O loop
    #         if W[i,j] ==0.:
    #             b_prob[i,j] = 0.
    #         else:

    #             for gamma in np.linspace(-200,0,5000):
    #                 gi = gamma * alpha / ((1-alpha) * sum_t_p_qt[i])
    #                 input_lamb= - W[i,j] * np.exp(1 + gi)

    #                 # if i==5:
    #                 #     print('----')
    #                 #     print('gamma',gamma) 
    #                 #     print('g_i',gi) 
    #                 #     print('input_lamb',  input_lamb)
    #                 #     print('lambert value',lambertw(input_lamb ,-1))
    #                 #     print('b prob', - W[i,j] / lambertw(input_lamb,-1 ))
                    

    #                 b_candidate = - W[i,j] / lambertw(input_lamb,-1 )
    #                 if  abs(b_candidate - 1.) <= abs(b_prob[i,j] - 1.):
    #                     b_prob[i,j] = b_candidate

                

    b_prob = np.zeros((num_states,len(symbols)))
    for i in range(len(all_hidden_states)): # Q loop
        b_candidate = np.zeros((len(symbols)))

        for gamma in np.linspace(-200,0,5000):
            for j in range(len(symbols)): # O loop
                
                if W[i,j] ==0.:
                    b_candidate[j] = 0.
                else:

                    gi = gamma * alpha / ((1-alpha) * sum_t_p_qt[i])
                    input_lamb= - W[i,j] * np.exp(1 + gi)
                    lamb = lambertw(input_lamb,-1 )
                    if np.iscomplex(lamb):
                        break
                    else:
                        b_candidate[j] = - W[i,j] / lamb

            if  abs(b_candidate.sum() - 1.) <= abs(b_prob[i,:].sum() - 1.):
                b_prob[i,:] = b_candidate





    b_prob = b_prob / b_prob.sum(axis=1)[...,None]
    # print(b_prob)
    assert b_prob.sum() == len(b_prob), "b_prob columns do not sum to 1"
    # import pdb;pdb.set_trace()



    # Computing A
    a_prob = np.zeros((num_states,num_states))
    for i,l in enumerate(all_hidden_states):
        for j,m in enumerate(all_hidden_states):
            a_prob[i, j] = A.get(l).prob(m)

    partials = np.zeros((T, num_states, num_states,num_states))
    for t in range(1, T):
        print(t)
        for i in range(len(all_hidden_states)):
            for l in range(len(all_hidden_states)):
                for m in range(len(all_hidden_states)):
                    if t==1:
                        if m==i:
                            partials[t,i,l,m] = p_qt[0,l]
                    else:
                        partials[t,i,l,m] = (partials[t-1,:,l,m]*a_prob[:,i]).sum()
                        if m==i:
                            partials[t,i,l,m] += p_qt[t-1, l]

    # import pdb;pdb.set_trace()

    # beta_l = np.zeros((num_states))
    # for l,l_v in enumerate(all_hidden_states):
    #     for val in transitions.get(l_v).values():
    #         beta_l[l] += val 
    # beta_l = alpha* beta_l

    denoms = np.zeros((num_states,num_states))
    numerators = np.zeros((num_states,num_states))
    a_prob = np.zeros((num_states,num_states))
    for l,l_v in enumerate(all_hidden_states):
        for m,m_v in enumerate(all_hidden_states):
            if transitions.get(l_v).get(m_v):
                numerators[l,m] = (-alpha*transitions.get(l_v).get(m_v))
                for t in range(T):
                    for i in range(len(all_hidden_states)):
                        for k in range(len(symbols)):
                            if b_prob[i,k] != 0:
                                denoms[l,m] += b_prob[i,k]*np.log(b_prob[i,k])*partials[t,i,l,m]
                denoms[l] *= (1-alpha)

    # l=-4
    for l,l_v in enumerate(all_hidden_states):
        for beta_l in np.linspace(-200,200,1000):
            
            a_candidate= numerators[l] / (denoms[l] + beta_l)
            # import pdb;pdb.set_trace()
            
            if  abs(a_candidate.sum() - 1.) <= abs(a_prob[l,:].sum() - 1.):
                # print(a_candidate)
                a_prob[l,:] = a_candidate

    a_prob = a_prob / a_prob.sum(axis=1)[...,None]
    assert a_prob.sum() == len(a_prob), "a_prob columns do not sum to 1"
    a_prob=np.abs(a_prob)
    import pdb;pdb.set_trace()


    return hmm.HiddenMarkovModelTagger(symbols, states, A, B, pi)



