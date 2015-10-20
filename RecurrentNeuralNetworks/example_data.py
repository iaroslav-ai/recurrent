'''
Created on Oct 20, 2015

@author: Iaroslav
'''

import numpy as np

def GenerateData(xsz, ysz, SeqSz, N, w):
    X,Y = [],[] # training data: set of sequences and desired outputs
    for i in range(N):
        x = np.random.randn(SeqSz, xsz);
        y = np.dot(x,w);
        y[1:] =y[1:] + y[0:-1] # add some dependency through time
        X.append(x)
        Y.append(y) 
    return X,Y