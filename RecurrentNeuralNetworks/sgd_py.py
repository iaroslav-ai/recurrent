'''
Created on Oct 16, 2015

@author: Iaroslav
'''

import numpy as np

# train network with sgd
def TrainNetwork(X,Y, network, alpha = 0.001, valchecks = 10, val_fraction = 0.3, C = 0.01):
    Wbest = np.copy( network.W )
    ValBest, checks = np.Inf, valchecks
    while checks > 0:
        # Training of the network for one epoch
        network.training = True;
        for i in range(int(val_fraction * len(X))):
            backprop = network.forward(X[i]) - Y[i]; 
            grad = network.backward(backprop) + C*network.W;
            network.W -= grad*alpha ;
        network.training = False;
        # validation of the network
        obj = 0
        for i in range(int(val_fraction * len(X)), len(X)):
            obj += 0.5*(np.linalg.norm( network.forward(X[i]) - Y[i], 2) ** 2); # L2 objective
        # update stopping criterion
        if obj < ValBest:
            ValBest , checks, alpha, Wbest = obj, valchecks, alpha*1.1, np.copy(network.W)
        else:
            checks, alpha = checks-1, alpha*0.7
        # output the training progress
        print "Validation error:", obj, "best:", ValBest
    network.W = Wbest