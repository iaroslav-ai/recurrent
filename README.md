This is a python implementation of various recurrent neural networks. The main goal was to minimize lines of code used for implementation, thus making code readable and maintainable, while preserving complete functionality needed for training and testing of the models. Does not requires any other dependencies other than numpy, and is mainly intended for prototyping. For example usage, see example.py .

Currently implemented are:

**Simple single hidden layer recurrent net.** Essentially a feedforward network, where the outputs of single hidden layer are connected to its inputs.

**Clockwork RNN.** Implementation of RNN described in http://arxiv.org/pdf/1402.3511v1.pdf, where only blocks with neighbouring clock rates can be connected. In the cited publication such RNN are shown to be superiour to LSTM RNN on some example datasets.