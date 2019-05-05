"""This is basic perceptron program which is used for binary classification"""

import random

class perceptron(object):
    """This perceptron class which is very basic neural network"""
    def __init__(self):
        """This is initialization of variables"""
        self.inputs = []
        self.TrainigInputs = []
        self.TrainigOutputs = []
        self.Weigths = [0]
        self.s = 0

    def training(self, tinput, toutput, iteraton=10000, learningRate=0.5,activationFn='binary'):
        """This is for training of perceptron this takes input set
        and output multiple input can be given for example if you want
        to train AND logic gate input can be given as [(0,0),(0,1),(1,0),(1,1)]
        and output can be given in form of [0,0,0,1]
        """


        self.inpt = []
        self.l =[1]
        """Following for loop for adding bias input in input tuple"""
        for i in tinput:
            if type(i) is int:
                l = [1]
                l.append(i)
                self.inpt.append(tuple(l))
            else:
                tin=[1]
                for k in i:
                    tin.append(k)
                self.inpt.append(tuple(tin))
        y = 0
        n = learningRate
        self.TrainigInputs = self.inpt
        self.TrainigOutputs = toutput

        if self.Weigths == [0]:
            """Generates initial random weights between 0 to 1"""
            self.Weigths = []
            for w in range(len(self.TrainigInputs[0])):
                self.Weigths.append(random.choice(range(2)))

        for it in range(iteraton):
            for i,d in zip(self.TrainigInputs,self.TrainigOutputs):
                s=0
                #Following is Summing Junction
                for x,w in zip(i,self.Weigths):
                    s = s + x*w
                y = self.activationFn(s,activationFn)
                error = d - y

                if error is not 0:
                    """weigth Update equation"""
                    for w in range(len(self.Weigths)):
                        self.Weigths[w] = self.Weigths[w] + (n*error*i[w])

    def activationFn(self,sum,functiontype='binary'):
        """this function selects activation functions here it is binary threshold is used.
        you can ad your choice of activation function by usign elif:
        i will be updating soon."""
        if functiontype is 'binary':

            if sum > 0:
                return 1
            else:
                return 0

    def run(self,input):
        """This is to run the trained neural network in this function first given input is
        checked and converted in tuples because our program is working on tuples
        as we have to attach biased input with every input for example you are checking
        and gate and your input is (0,1) this is converted to (1,0,1) with bias added and if you are
        checking for multiple inputs your input will be [(0,1),(1,1)] this will converted to
        [(1,0,1),(1,1,1)] and if you are checking single input gate like NOT then if your isn put
        is (1) then it will converted to (1,1) and if you are using multiple single input like [1,0,1]
        the that will converted to [(1,1),(1,0),(1,1)]"""
        inpt =[]
        output = []
        if type(input) is int:
            ib = [1]
            ib.append(input)
            inpt.append(tuple(ib))
        else:
            if type(input) is tuple:
                tin = [1]
                for i in input:
                    tin.append(i)
                inpt.append(tuple(tin))
            else:
                tin = [1]
                for k in input:

                    if type(k) is int:
                        tin=[1]
                        tin.append(k)
                        inpt.append(tuple(tin))
                    else:
                        tin = [1]
                        tin.extend(k)
                        inpt.append(tuple(tin))
        """From following point it just sum up inputs and generate output using 
        Trained/optimized weights"""

        for x in inpt:
            s=0
            #Summing Junction
            for i,w in zip(x,self.Weigths):
                s = s+i*w
            output.append(self.activationFn(s))
        return output
"""Following example for OR logic Gate
inp= [(0,0),(0,1),(1,0),(1,1)]
op = [0,1,1,1]
orGate = perceptron()
orGate.training(inp,op)
print("this are wights",orGate.Weigths)

print("Result",orGate.run((1,1))) #Testing for single input (1,1)
print("Result",orGate.run([(1,1),(0,0)])) #Testing for multiple input (1,1) and (0,0)"""




