"""A simple Fruit Classification Problem
Objective: Train perceptron Neural Net using given 4 fruit data
and classify unknown fruit. input is in form (a,b) where a is weights
in gms and b is Length in in cms and output is 1 for class 1 and 0 for class 2
given data is Class 1 Data
              |(121,16.8)  |  1 |
              |(114,15.2)  |  1 |
              Class 2 Data
              |(210,9.4)   |  0 |
              |(195,8.1)   |  0 |
              and unknown fruit data (140,17.9)"""

from Nperceptron import *


fruit = perceptron() #define classifier named fruit
Inp = [(121,16.8),(114,15.2),(210,9.4),(195,8.1)]
op = [1,1,0,0]

fruit.training(Inp,op,100000,20)# training fruit classifier
print(fruit.Weigths)
a = fruit.run((140,17.9)) #classifying unknown fruit using trained NN
if a is 1:
    print("Given Fruit is belongs to Class 1")
else:
    print("Given Fruit is belongs to Class 2")
