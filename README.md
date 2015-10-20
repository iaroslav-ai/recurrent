Here I experiment with Gaussian RNN and compare it to RNN consisting of neurons with other activation functions.

A boosting - like approach is used to train RNN: weights of neurons are selected randomly, and only the output weights are modified. This allows to reduce training to **convex** optimization, which means that solution that I get is always the best possible.