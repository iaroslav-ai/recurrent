'''
Created on Oct 16, 2015

@author: Iaroslav
'''

import numpy as np
# A variation of clockwork RNN, as in http://arxiv.org/pdf/1402.3511v1.pdf
# Here for simplicity only blocks with neighboring clock rates are connected 
class CWRNN:
    training = False; # collect data for gradient computation when true
    def __init__(self, xsz, ysz, blocks, neurons_per_block):
        self.W = np.random.randn(neurons_per_block*2+xsz+ysz+1, neurons_per_block*blocks)*0.001
        self.xsz = xsz
        self.ysz = ysz
        self.blocks = blocks
        self.npb = neurons_per_block
    # compute outputs for given sequence x
    def forward(self,x):
        netinput = self.W[:-self.ysz,0]*0; 
        netinput[-1] = 1; # bias constant
        Hs = np.zeros(self.npb)
        self.activations, self.inputs = [], []
        for b in range(self.blocks):
            self.activations.append([])
            self.inputs.append([])
        y = np.zeros((x.shape[0], self.ysz)) # outputs
        for i in range(x.shape[0]):
            # format: block outputs (H) | slower block outputs (Hs) | x | bias
            netinput[self.npb*2:-1] = x[i];
            y[i] = 0
            for b in range(self.blocks):
                H = np.maximum( np.dot(netinput,self.W[0:-self.ysz, b*self.npb:(b+1)*self.npb]) , 0)
                if self.training:
                    self.activations[b].append(np.copy(H));
                    self.inputs[b].append(np.copy(netinput));
                netinput[:self.npb] = H;
                netinput[self.npb:self.npb*2] = H * (i % 2 ** b == 0);
                y[i] += np.dot(self.W[-self.ysz:, b*self.npb:(b+1)*self.npb]  , H)
        return y
    # compute gradient of the network; assumes that "forward" was called before
    def backward(self,backprop):
        grad = np.copy( self.W )*0;
        bptt = self.W[0,]*0; # this is responsible for backprop through time
        for i in range(len(self.activations)-1,-1,-1):
            bptb = np.zeros(self.npb) # backprop through blocks
            for b in range(self.blocks-1,-1,-1):
                H = self.activations[b][i]
                grad[-self.ysz:, b*self.npb:(b+1)*self.npb] += np.outer( backprop[i,], H )
                bck = np.dot( backprop[i,], self.W[-self.ysz:,b*self.npb:(b+1)*self.npb] )
                bck = (bck + bptt[b*self.npb:(b+1)*self.npb] + bptb* (i % 2 ** b == 0)) * (H > 0);
                grad[0:-self.ysz,b*self.npb:(b+1)*self.npb] += np.outer(self.inputs[b][i], bck)
                bptt[b*self.npb:(b+1)*self.npb] = np.dot( self.W[:self.npb,b*self.npb:(b+1)*self.npb], bck )
                bptb = np.dot( self.W[self.npb:self.npb*2,b*self.npb:(b+1)*self.npb], bck )
        return grad