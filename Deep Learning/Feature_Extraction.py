#!/usr/bin/env python
# coding: utf-8

# ## Speech and Speaker Recognition - DT2119 VT19-1 
# 
# ### Feature Extraction - Lab 1

# In[1]:


from __future__ import print_function

import numpy as np
import math

from lab1_tools import lifter,trfbank
from lab1_tools import trfbank
from lab1_tools import tidigit2labels

from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from sklearn.mixture import GaussianMixture
from scipy.fftpack.realtransforms import dct
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage,dendrogram

from matplotlib import pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')




# In[4]:


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    
    no_windows_to_fit = float(np.abs(len(samples) - winlen)) / winshift
    N = int(np.ceil(no_windows_to_fit))
    frames = np.empty((N, winlen))
    
    for i in range(frames.shape[0]):
        n = i * winshift
        frames[i,:] = samples[n:n+winlen]
    
    return frames


#  ###### Sample the data into smaller frames 
#  
#  Given winlen = 20 milli seconds
#        winshift = 10 milli seconds       
#  Sampling rate = 20,000  
#  i.e 20000 frames per second or 1000 milli seconds  
#  i.e we have 20 frames for each micro second  
#  hence the winlen and winshift in similar units of samples would be   
#  20*winlen(ms) and 20*winshift(ms)  
#  Giving us winlen = 400 and winshift = 200  

# In[5]:




# In[6]:

# ### 4.2  Pre-emphasis

# ###### increases the amplitude of high frequency bands and decrease the amplitudes of lower bands. Noise handling!
# From lecture what pre-emph need to do  
# ##### y[n] = x[n] − αx[n − 1]  
# what lfilter does  
# scipy.signal.lfilter(b, a, x, axis=-1, zi=None)  
# ##### a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]           - a[1]*y[n-1] - ... - a[N]*y[n-N]  
# Here b would be 1,-alpha  
# a would be 1 and a1-n would be 0  
# x would be the frame  

# In[7]:


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """    
    b = [1,-p]
    a = [1]
    return lfilter(b, a, input)


# In[8]:


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    n_frames = input.shape[1]
    hamming_window = hamming(n_frames, sym=False)
   
    #uncomment to display window
    #_ = plt.rcParams['figure.figsize'] = [5, 3]
    #plt.plot(hamming_window)
    #plt.show()
    
    windowed = input * hamming_window
    return windowed


# In[10]:


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    fftransform = fft(input, nfft)
    spec = np.square(np.absolute(fftransform))
    return spec


# In[12]:





def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    
    filter_bank = trfbank(samplingrate, input.shape[1])
    #uncomment to display filter bank plots
    
#     _ = plt.rcParams['figure.figsize'] = [15, 3]
#     _ = plt.subplot(211)
#     _ = plt.plot(filter_bank)
#     _ = plt.subplot(212)
#     _ = plt.plot(filter_bank.T)
#     plt.show()

    #too much variation from the example
    #without doing this there were only handful of variations from example
    #filter_bank = np.where(filter_bank == 0, np.finfo(float).eps, filter_bank)  # Numerical Stability
    return np.log(input.dot(filter_bank.T))


# In[15]:


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform
    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    mfcc = dct(input)[:,0:nceps]
    return mfcc


# In[16]:


# Function given by the exercise ----------------------------------
def mspecFn(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


# In[19]:


def mfccFn(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspec = mspecFn(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)


# In[20]:




def dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1)):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N, M = len(x), len(y)
    
    LD = np.zeros((N + 1, M + 1))
    LD[0, 1:] = np.inf
    LD[1:, 0] = np.inf
    
    for i in range(N):
        for j in range(M):
            LD[i+1][j+1] = dist(x[i], y[j])
    
    AD = LD
    
    for i in range(N):
        for j in range(M):
            AD[i+1][j+1] += min(AD[i, j], AD[i, j+1], AD[i+1, j])

    path = [[N,M]]
    
    i, j = N, M
    while (i > 0 and j > 0):
        path_min = np.argmin((AD[i-1, j-1], AD[i-1, j], AD[i, j-1]))

        if (path_min == 0):
            i = i - 1
            j = j - 1
        elif (path_min == 1):
            i = i - 1
        elif (path_min == 2):
            j = j - 1

        path.append([i,j])

    LD = LD[1:, 1:]
    AD = AD[1:, 1:]

    d = AD[-1, -1]/(N+M)
    
    return d, LD, AD, path


# In[73]:


#dist = lambda x, y: cdist(x.reshape(1,-1) , y.reshape(1,-1))



# global_dist, LD, AD, path = dtw(mfccs, mfccs)#,np.linalg.norm)
# print(global_dist.shape)
# D = global_dist
# for i in xrange(len(D)):
#     for j in xrange(len(D)):
#         D[i,j] = global_dist

# mfccs_1 = mfccs[0]
# mfccs_2 = mfccs[1]
# i, j = 5, 6
# route = np.zeros_like(AD)
# global_dist, LD, AD, path = dtw(mfccs[i], mfccs[j])#,np.linalg.norm)

# for i in path:
#     route[(i)] = 150
# _ = plt.rcParams['figure.figsize'] = [15, 4]
# _ = plt.subplot(131).set_title("Loc")
# _ = plt.pcolormesh(LD)
# _ = plt.subplot(132).set_title("Acc")
# _ = plt.pcolormesh(AD)
# _ = plt.subplot(133).set_title("Route")
# _ = plt.pcolormesh(route)
# _ = plt.show()


# ##### References :  
# 1. (2019). Download.ni.com. Retrieved 20 April 2019, from http://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
# 2. Fayek, H. (2016). Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What’s In-Between. Haytham Fayek. Retrieved 21 April 2019, from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
