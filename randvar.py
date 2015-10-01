
import numpy as np
from matplotlib import pyplot as plt
from types import *
from scipy.stats import rv_discrete



class FiniteDiscreteRandomVariable(rv_discrete):
    '''implements an discrete random variable with finite support.
    
    members:
    
    name: a string that holds a human-readable name for the variable.
    support:  a list containing the support of the variable
    pmf(x): the probability mass function, returns a float.
    E(f = lambda x:x): computes the expectation of a function f. Default computes expectation of RV.
    H(): returns entropy of RV.
    '''

    def __init__(self,support = [0], pmf = None, name = None):
        self.support = support

        
        self.name = name
        if(hasattr(pmf,'__call__')):
            self.data = {x:pmf(x) for x in support}        
        else:
            self.data = {x:pmf[x] for x in support}

        if(pmf==None):
            self.data = {x:1.0/len(support) for x in support}
        
        Z = np.sum(self.data.values())
        if(Z != 1.0):
            for x in support:
                self.data[x] = self.data[x] / Z

        if(name == None):
            self.name = 'Unknown RV'

        super(FiniteDiscreteRandomVariable,self).__init__(name = name,values = (self.data.keys(),self.data.values()))

    def sample(self):
        '''returns a sample of this RV'''
        
    def __str__(self):
        return self.name
    def support(self):
        return self.data.keys()

    def printpmf(self):
        print self.data

    def E(self,f = lambda x:x):
        return self.expect(f)
        #return np.sum([self.pmf(x)*f(x) for x in self.support])

    def H(self):
        return self.entropy()
        #return self.E(lambda x:-np.log(self.pmf(x)) if self.pmf(x)!=0 else 0)
        
    def add(self,other):
        '''creates RV that is the sum of two RVs, assumes both RVs to be summed are independent.'''
        if(type(other)==type(self)):
            return BinaryOperation(self,other,op=lambda x,y:x+y,name = self.name+" + "+other.name)
        else: #assume that other is a constant number
            return Operation(self,lambda x:x+other,name=self.name+ " + "+str(other))

    def mul(self,other):
        '''creates RV that is the product of two independent RVs'''
        if(type(other)==type(self)):
            return BinaryOperation(self,other,op=lambda x,y:x*y,name = self.name+" * "+other.name)
        else: 
            return Operation(self,lambda x:x*other,name=str(other)+" * "+self.name)

    def __add__(other,self):
        return self.add(other)
    def __mul__(other,self):
        return self.mul(other)

    def __add__(self,other):
        return self.add(other)
    def __mul__(self,other):
        return self.mul(other)



def Operation(RV,op,name=None):
    '''creates a RV that is the result of op(RV)'''
    support = np.unique([op(x) for x in RV.support])
    pmf = {}
    for x in RV.support:
        if(op(x) not in pmf):
            pmf[op(x)]=0
        pmf[op(x)]+=RV.pmf(x)
    return FiniteDiscreteRandomVariable(support = support, pmf = pmf, name = name)

def BinaryOperation(RV1,RV2,op,name=None):
    '''creates a RV that is the result of op(RV1,RV2), assuming RV1 and RV2 are independent'''

    support = np.unique([op(x,y) for x in RV1.support for y in RV2.support])
    pmf = {}
    for x in RV1.support:
        for y in RV2.support:
            if op(x,y) not in pmf:
                pmf[op(x,y)]=0
            pmf[op(x,y)] += RV1.pmf(x)*RV2.pmf(y)
    return FiniteDiscreteRandomVariable(support = support,pmf = pmf,name = name)
