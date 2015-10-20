'''
Created on Oct 16, 2015

@author: Iaroslav
'''

import numpy as np
# Simple RNN
class RNN:
    training = False; # collect data for gradient computation when true
    def __init__(self, xsz, ysz, neurons_per_block):
        self.W = np.random.randn(neurons_per_block+xsz+ysz+1, neurons_per_block)*0.001
        self.xsz = xsz
        self.ysz = ysz
    # compute outputs for given sequence x
    def forward(self,x):
        netinput = self.W[:-self.ysz,0]*0;
        netinput[-1] = 1;
        self.activations, self.inputs = [], []
        y = np.zeros((x.shape[0], self.ysz)) # outputs
        for i in range(x.shape[0]):
            netinput[self.W.shape[1]:-1] = x[i];
            H = np.maximum( np.dot(netinput,self.W[0:-self.ysz,]) , 0)
            if self.training:
                self.activations.append(np.copy(H));
                self.inputs.append(np.copy(netinput));
            netinput[:self.W.shape[1]] = H;
            y[i] = np.dot(self.W[-self.ysz:,]  , H)
        return y
    # compute gradient of the network; assumes that "forward" was called before
    def backward(self,backprop):
        grad = np.copy( self.W )*0;
        bptt = self.W[0,]*0; # this is responsible for backprop through time
        for i in range(len(self.activations)-1,-1,-1):
            H = self.activations[i]
            grad[-self.ysz:,] += np.outer( backprop[i,], H )
            bck = np.dot( backprop[i,], self.W[-self.ysz:,] )
            bck = (bck + bptt) * (H > 0);
            grad[0:-self.ysz,] += np.outer(self.inputs[i], bck)
            bptt = np.dot( self.W[:self.W.shape[1],], bck )
        return grad