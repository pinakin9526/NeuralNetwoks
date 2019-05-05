# NeuralNetwoks
This repository contains python library for basic neural nets like perceptron, feed forward neural Nets

Nperceptron.py contains basic perceptron NN with binary activation function.

second file NNtest.py contains one example for training Perceptron NN for fruit classification
and using that classifier you can classify unkown fruit in class 1 or class 2 acording to its 
physical property like weight and length.

**Usage**:

### For Example you want to Train NN to mimic Logic AND GATE you can do following

```python

from Nperceptron import * 
inpt = [(0,0),(0,1),(1,0),(1,1)]         # this is how you create training input data set
top = [0,0,0,1]                          # this is called tragate data
lANDgate = perceptron()                  #this is how you define neural net variable
lANDgate.training(inpt,top,10000,5,'binary')  #this is training function which takes five arguments Training Input,
                                              #Training Outputs,No of Iteration,Learning rate,Activation Function
"""NOw you can test your trained neural net by following code"""
a = lANDgate.run((0,1))        #for single input (0,1). result will be stored in a
print(a)                       # to display result which is 0
b = lANDgate.run([(0,1),(1,1)] #this is how you can give multiple input
print(b)                       # to display result which is [0,1]

```
