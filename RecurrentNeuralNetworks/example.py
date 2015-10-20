'''
Created on Oct 4, 2015

@author: Iaroslav
'''

import numpy as np
from rnn_py import RNN
from cwrnn_py import CWRNN
from sgd_py import TrainNetwork
from example_data import GenerateData

xsz, ysz, SeqSz, N = 3, 3, 5, 2000

# generate random parameters of ground truth
w = np.random.randn(xsz, ysz);
X,Y = GenerateData(xsz, ysz, SeqSz, N, w)
Xt,Yt = GenerateData(xsz, ysz, SeqSz, 1, w)

#rnn = RNN(xsz, ysz, 10)
rnn = CWRNN(xsz, ysz, 2, 10)

TrainNetwork(X,Y, rnn);

print "original sequence: "
print  Yt[0]
print "predicted sequence: "
print  rnn.forward(Xt[0])