'''
Created on Oct 4, 2015

@author: Iaroslav
'''

import numpy as np
from rnn_py import RNN
from sgd_py import TrainNetwork

xsz, ysz, SeqSz, N, neurons = 2, 2, 5, 500, 10
w = np.random.randn(xsz, ysz);
def GenerateData(xsz, ysz, SeqSz, N):
    X,Y = [],[] # training data: set of sequences and desired outputs
    for i in range(N):
        x = np.random.randn(SeqSz, xsz);
        y = np.dot(x,w);
        y[1:] =y[1:] + y[0:-1] # add some dependency through time
        X.append(x)
        Y.append(y) 
    return X,Y

X,Y = GenerateData(xsz, ysz, SeqSz, N)
Xt,Yt = GenerateData(xsz, ysz, SeqSz, 1)

rnn = RNN(xsz, ysz, neurons)

TrainNetwork(X,Y, rnn);

print "original sequence: "
print  Yt[0]
print "predicted sequence: "
print  rnn.forward(Xt[0])