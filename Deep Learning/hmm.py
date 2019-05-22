#!/usr/bin/env python
# coding: utf-8

# ## Speech and Speaker Recognition - DT2119 VT19-1 
# 
# ### HMM - Lab 2

# In[1]:


from __future__ import print_function

import numpy as np
import math

from lab2_tools import *

import prondict as prondict

from matplotlib import pyplot as plt
import seaborn as sns

import warnings
get_ipython().magic(u'matplotlib inline')
from tqdm import tqdm



# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


# In[10]:


def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    concatedHMM = {}
    #M is the number of emitting states in each HMM model (could be different for each)
    #K is the sum of the number of emitting states from the input models
    
    M1 = hmm1['means'].shape[0]
    M2 = hmm2['means'].shape[0]
    K = M1 + M2
    
    concatedHMM['name'] = hmm1['name'] + hmm2['name']
    concatedHMM['startprob'] = np.zeros((K + 1, 1))
    concatedHMM['transmat'] = np.zeros((K + 1, K + 1))
    concatedHMM['means'] = np.vstack((hmm1['means'],hmm2['means']))
    concatedHMM['covars'] = np.vstack((hmm1['covars'],hmm2['covars']))
        
    
    start1 = hmm1['startprob'].reshape(-1,1)
    start2 = hmm2['startprob'].reshape(-1,1)
    
    concatedHMM['startprob'][:hmm1['startprob'].shape[0]-1,:] = start1[:-1,:]
    concatedHMM['startprob'][hmm1['startprob'].shape[0]-1:,:] = np.dot(start1[-1,0],start2)
    trans = concatedHMM['transmat']
    trans1 = hmm1['transmat']
    trans2 = hmm2['transmat']

    trans[:trans1.shape[0]-1,:trans1.shape[1]-1] = trans1[:-1,:-1]
    temp = trans1[:-1,-1].reshape(-1,1)
    trans[:trans1.shape[0]-1,trans1.shape[1]-1:] =                             np.dot(temp,start2.T)
    trans[trans1.shape[0]-1:,trans1.shape[1]-1:] = trans2
    concatedHMM['transmat'] = trans    
    
    return concatedHMM


# ### 4.1 Example

# In[11]:



# ### 5.1 Gaussian emission probabilities

# In[12]:


# Here we plot the probablities of each state given an instance in the utterance of the sound.  
# And we also know that each phoneme's model is made of 3 states. So the word model on O is made of 3 phoneme models of O(ow) and 2 sil's in the start and end. So the first and last frames of the sample are mostly silent and the middle frames correspond to the word 'O'
# 
# From the color bar blue is negative 700 i.e the pobability that the model is that state given the corresponding frame is lower and higher when its yellow.
# 
# Thus as expected the start and end of the sample is more yellow in the states 1,2,3 and 7,8,9 and yellow in the middle in states of 4,5,6 which is the word 'O'

# ### 5.2 Forward Algorithm

# In[13]:


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    alpha = np.zeros(log_emlik.shape)
    alpha[0][:] = log_startprob.T + log_emlik[0]

    for n in range(1,len(alpha)):
        for i in range(alpha.shape[1]):
            alpha[n, i] = logsumexp(alpha[n - 1] + log_transmat[:,i]) + log_emlik[n,i]
    return alpha


# In[14]:

# Forward Pass: The probabilty of observing a given sequence till that given time being in a particular state.
# In other words 
# ##### probability of being in a state given a sequence
# 
# As given 1 sample, the model can obviously can not reach state 2 or ...9 hence their probabilty is 0 whose log is -inf and therefore not displayed on colormap with the warning divide by zero.
# 
# So from the observation likelihood calculated in the previous section, we observed that the utterance of O lies in the frames 10-50 after which 2nd silence would start. This alpha pass shows that same with higher values in 4-6 states in 10-50 frames. Also as the frames move to end the probabilities of states tend to increase as we move from state 1 to 9.
# 
# Note that the comparision of the probabilites between the frames has to be seen w.r.t to the particular frame and not in the whole colormesh.

# #### the likelihood P(X|θ) of the whole sequence X = {x0, . . . , xN−1}, given the model parameters θ 
# As discussed before, alpha is the probabilty of observing a given sequence till that given time being in a particular state. 
# Hence the total probabilty of obeserving such a sequence with the given model is the sum of probabilties to observe the given sequence over all the possible states. 

# In[15]:



# In[16]:



# In[17]:




# ###### Complexity:
# CPU times: user 20.9 s, sys: 28 ms, total: 21 s   
# Wall time: 20.9 s
# ###### Accuracy
# One Speaker - 77.27% 
# All Speakers - 97.72%

# ### 5.3 Viterbi Approximation
# ##### finding the most likely sequence of hidden states

# In[20]:


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    
    N = log_emlik.shape[0]
    M = log_emlik.shape[1]
    
    viterbi_path = np.empty((N), dtype=np.int)
    viterbi_loglik = 0
    V = np.zeros((N,M))
    B = np.zeros((N,M))
    
    for j in range(M):
        V[0, j] = log_startprob[j] + log_emlik[0,j]
        
    for n in range(1,N):
        for j in range(M):
            V[n, j] = np.max(V[n-1,:] + log_transmat[:,j]) + log_emlik[n,j]
            B[n, j] = np.argmax(V[n-1,:] + log_transmat[:,j])
    
    viterbi_path[-1] = np.argmax(V[-1, :])
    viterbi_loglik = V[N-1, viterbi_path[-1]]
    
    for n in range(0,N-1):
        viterbi_path[n] += np.max(V[n-1,:])
    
    for n in reversed(range(N-1)):
        for j in range(M):
            viterbi_path[n] = B[n+1, viterbi_path[n+1]]
            
    return (viterbi_loglik, viterbi_path)





# ###### Complexity:
# CPU times: user 7.51 s, sys: 4 ms, total: 7.52 s
# Wall time: 7.51 s
# ###### Accuracy
# One Speaker - 77.27% 
# All Speakers - 100%
# 
# Viterbi faster than alpha pass/forward
# Accuracies are same for one speaker   
# For all speakers viterbi performs perfect   
# but alpha have some wrong predictions

# ### 5.4 Backward Algorithm
# ###### Probability of the next observation given a state

# In[27]:


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

    N, M = log_emlik.shape
    beta = np.empty((N,M))
    beta[-1,:] = 0
    for n in reversed(range(N-1)):
        for i in range(M):
            beta[n,i] = logsumexp(log_transmat[i,:] + log_emlik[n+1,:] + beta[n+1,:])
    return beta


# In[28]:



# ### 6 HMM Retraining (emission probability distributions)

# ### 6.1 State posterior probabilities / Gamma
# 
# ###### Given the entire observation sequence and current estimate of the HMM, what is the probability that at time (t) the hidden state is (Xt=i)

# In[29]:


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N = len(log_alpha)
    log_gamma = log_alpha + log_beta - logsumexp(log_alpha[N - 1])
    return log_gamma



# In[37]:




# In[31]:


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    gamma = np.exp(log_gamma)
    means = np.zeros((log_gamma.shape[1], X.shape[1]))
    covars = np.zeros(means.shape)

    for i in range(means.shape[0]):
        gamma_sum = np.sum(gamma[:,i])
        means[i] = np.sum(gamma[:,i].reshape(-1, 1) * X, axis = 0) / gamma_sum
        covars[i] = np.sum(gamma[:,i].reshape(-1, 1) * (X - means[i])**2, axis = 0) / gamma_sum
        covars[i, covars[i] < varianceFloor] = varianceFloor
    return (means, covars)


# In[128]:


# Model -4 itself
# 
# 25%|██▌       | 5/20 [00:00<00:01,  9.08it/s]  
# last iteration update, prev -5994.049059632479 -5994.049059721665  
# [-6826.654332902908, -6154.595775649375, -6022.524852217402, -5998.1572849760105, -5994.049059721665]

# Model - o
# 
# 20%|██        | 4/20 [00:00<00:01, 13.95it/s]
# last iteration update, prev -6318.813712693775 -6318.817002586534
# [-7070.173641433824, -6451.2981231871045, -6352.529477038947, -6321.501698999799, -6318.817002586534]
# 
# On model - 6
# 
# 25%|██▌       | 5/20 [00:00<00:01,  7.52it/s]  
# last iteration update, prev -5884.1937370000305 -5884.590499120586  
# [-7193.099605598604, -6146.134276859259, -5934.147135682102, -5890.096108797846, -5884.590499120586]  
