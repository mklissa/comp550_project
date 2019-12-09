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

import random
import time
np.random.seed(1)
random.seed(1)
from scipy.special import xlogy
import warnings
warnings.filterwarnings('ignore')

def train_supervised(states, symbols, labelled_sequences, estimator=None, extra_sents=None, mihmm=False):

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

    if mihmm:
        alpha=0.3
        T=400

        all_hidden_states = A.conditions()
        num_states = len(all_hidden_states)
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
       

        ### Not optimzed code ###
        # start=time.time()
        # b_prob = np.zeros((num_states,len(symbols)))
        # for i in range(len(all_hidden_states)): # Q loop
        # # i=-3
        #     b_candidate = np.zeros((len(symbols)))

        #     for gamma in np.linspace(-100,0,5000):
        #         for j in range(len(symbols)): # O loop
                    
        #             if W[i,j] ==0.:
        #                 b_candidate[j] = 0.
        #             else:
                        
        #                 gi = gamma  / ((1-alpha) * sum_t_p_qt[i])
        #                 input_lamb= - W[i,j] * np.exp(1 + gi)
        #                 lamb = lambertw(input_lamb,-1 )
        #                 if np.iscomplex(lamb) or np.isinf(lamb):
        #                     break
        #                 else:
        #                     # print(gamma,lamb,- W[i,j] / lamb)
        #                     b_candidate[j] = - W[i,j] / lamb

        #         if  abs(b_candidate.sum() - 1.) <= abs(b_prob[i,:].sum() - 1.):
        #             # print(gamma,b_candidate.sum(),lamb)             
        #             b_prob[i,:] = b_candidate
        # print(time.time()-start)
        # import pdb;pdb.set_trace()
        ### Not optimzed code ###



        b_prob = np.zeros((num_states,len(symbols)))
        for i in range(len(all_hidden_states)): # Q loop
        # i=0
            b_candidate = np.zeros((len(symbols)))
            non_zeros = (W[i] != 0.)

            for gamma in np.linspace(-4000,0,5000): #for gamma in np.linspace(-150,0,3000):
                gi = gamma  / ((1-alpha) * sum_t_p_qt[i])
                input_lamb= - W[i,:] * np.exp(1 + gi)
                lamb = lambertw(input_lamb,-1 )
                if np.iscomplex(lamb[non_zeros]).sum() or np.isinf(lamb[non_zeros]).sum():
                    continue
                else:
                    # print(gamma,lamb,- W[i,j] / lamb)
                    # import pdb;pdb.set_trace()
                    b_candidate[non_zeros] = - W[i,non_zeros] / lamb[non_zeros]

                if  abs(b_candidate.sum() - 1.) <= abs(b_prob[i,:].sum() - 1.):
                    # print(gamma,b_candidate.sum())               
                    b_prob[i,:] = b_candidate
                    best_gamma=gamma
            print(best_gamma)



        # orig_b_prob = np.zeros((num_states,len(symbols)))
        # for q,i in enumerate(all_hidden_states): # Q loop
        #     for o,j in enumerate(symbols): # O loop
        #         orig_b_prob[q, o] = B.get(i).prob(j)


        print('b_prob',b_prob.sum(axis=1))
        b_prob = b_prob / b_prob.sum(axis=1)[...,None]
        # assert b_prob.sum() == len(b_prob), "b_prob columns do not sum to 1"

        # Computing A
        a_prob = np.zeros((num_states,num_states))
        for i,l in enumerate(all_hidden_states):
            for j,m in enumerate(all_hidden_states):
                a_prob[i, j] = A.get(l).prob(m)


        ### Not optimzed code ###
        # start=time.time()
        # partials = np.zeros((T, num_states, num_states, num_states))
        # for t in range(1, T):
        #     # print(t)
        #     for i in range(num_states):
        #         for m in range(num_states):
        #             for l in range(num_states):
        #                 if t==1:
        #                     if m==i:
        #                         partials[t,i,l,m] = p_qt[0,l]
        #                 else:
        #                     partials[t,i,l,m] = (partials[t-1,:,l,m]*a_prob[:,i]).sum()
        #                     if m==i:
        #                         partials[t,i,l,m] += p_qt[t-1, l]
        ### Not optimzed code ###


        partials = np.zeros((T, num_states, num_states, num_states))
        for t in range(1, T):
            for i in range(num_states):
                for m in range(num_states):
                    if t==1:
                        if m==i:
                            partials[t,i,:,m] = p_qt[0,:num_states]
                    else:
                        partials[t,i,:,m] = (partials[t-1,:,:,m]*a_prob[:num_states,i][:,None]).sum(axis=0)
                        if m==i:
                            partials[t,i,:,m] += p_qt[t-1,:]


        
        

        denoms = np.zeros((num_states,num_states))
        numerators = np.zeros((num_states,num_states))
        a_prob = np.zeros((num_states,num_states))

        ### Not optimzed code ###
        # for l,l_v in enumerate(all_hidden_states):
        #     for m,m_v in enumerate(all_hidden_states):
        #         if transitions.get(l_v).get(m_v):
        #             numerators[l,m] = (-alpha*transitions.get(l_v).get(m_v))
        #             for t in range(T):
        #                 for i in range(len(all_hidden_states)):
        #                     for k in range(len(symbols)):
        #                         if b_prob[i,k] != 0:
        #                             denoms[l,m] += b_prob[i,k]*np.log(b_prob[i,k])*partials[t,i,l,m]
        #     denoms[l] *= (1-alpha)
        ### Not optimzed code ###




        for l,l_v in enumerate(all_hidden_states):
            for m,m_v in enumerate(all_hidden_states):
                
                if transitions.get(l_v).get(m_v):
                    numerators[l,m] = (-alpha*transitions.get(l_v).get(m_v))
                    denoms[l,m]= (xlogy(b_prob,b_prob)*partials.sum(axis=0)[:,l,m][:,None]).sum()
            denoms[l] *= (1-alpha)        



        for l in range(num_states):
        # l=0
            for beta_l in np.linspace(-500,40000,80000):
                
                a_candidate= numerators[l] / (denoms[l] + beta_l)
                
                if  abs(a_candidate.sum() - 1.) <= abs(a_prob[l,:].sum() - 1.):
                    # print(beta_l,a_candidate.sum())
                    a_prob[l,:] = a_candidate
                    best_beta=beta_l
            print(best_beta)

        print('a_prob',a_prob.sum(axis=1))
        # import pdb;pdb.set_trace()
        a_prob = a_prob / a_prob.sum(axis=1)[...,None]
        # assert a_prob.sum() == len(a_prob), "a_prob columns do not sum to 1"
        a_prob=np.abs(a_prob)
        


        for i,l in enumerate(all_hidden_states):
            for j,m in enumerate(all_hidden_states):
                A.get(l)._freqdist[m] = a_prob[i, j]

        for q,i in enumerate(all_hidden_states): 
            for o,j in enumerate(symbols): 
                B.get(i)._freqdist[j] = b_prob[q, o]




    return hmm.HiddenMarkovModelTagger(symbols, states, A, B, pi)



