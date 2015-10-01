
import numpy as np
from matplotlib import pyplot as plt
from types import *
import randvar as RV
from copy import deepcopy

class Node(object):
    '''
    a node in a graphical model.
    all our graphical models will represent exponential families.
    random variables will be finite discrete real RVs.
    an edge between two nodes X and Y indicates that XY is a sufficient statistic.
    All nodes are themselves sufficient statistics.
    There are no other sufficient statistics.

    support: support of this node's RV
    theta: coefficient of this node in exp. family.
    name: human-readable name of node
    edges: list of edges connected to this node
    incoming_messages: messages we've received from factor nodes
    adjacent: list of adjacent factor nodes.
    pending: list of factor nodes we need to send a new message to.
              a factor node N is pending if we've received a message from a different 
              factor since we last updated N
    '''

    def __init__(self,support=[0],name=None,theta=0):
        self.support = support
        self.observedsupport = deepcopy(support)
        self.theta = theta
        self.name = name
        self.edges = []

        self.incoming_messages = {}
        self.adjacent = []
        self.pending = set([])

    def getmessage(self,factor,message,debug):
        '''
        gets a message from a factor. debug causes printing for when this didn't work yet.
        '''
        if(debug):
            print "node "+str(self)+ " got message from factor "+str(factor)
        self.incoming_messages[factor] = message
        addpending = set(self.adjacent)
        addpending.remove(factor)
        self.pending = addpending.union(self.pending)

    def sendmessages(self,debug = False):
        '''
        sends all pending messages.
        '''
        if(len(self.pending)==0):
            return False
        for factor in self.pending:
            message = self.computemessage(factor,debug)
            factor.getmessage(self,message,debug)
        self.pending = set([])
        return True

    def computemessage(self,factor,debug):
        '''
        computes the message that goes to a particular factor.
        '''
        message = {}
        for x in self.observedsupport:
            m = 1
            for f in self.adjacent:
                if(f==factor):
                    continue
                try:
                    m *= self.incoming_messages[f][x]
                except KeyError:
                    pass

            message[x] = m
        if(debug):
            print "message from node "+self.name+" to factor "+str(factor)+":"
            print "\t"+str(message)
        return message        

    def __str__(self):
        return self.name
    
    def marginal(self):
        '''
        returns a FiniteDiscreteRandomVariable describing the marginal for this node.
        '''
        pmf ={z:reduce(lambda x,y:x*y,[self.incoming_messages[f][z] for f in self.adjacent]) for z in self.observedsupport}
        return RV.FiniteDiscreteRandomVariable(self.observedsupport,pmf,self.name)



class Edge(object):
    '''
    an edge in a graphical model

    node1: first node in edge
    node2: second node in edge
    theta: coefficient of node1*node2 in exp. family.
    name: a name for this edge name has form (node1, node2)
    '''

    def __init__(self,nodes,theta= 0):
        self.nodes = nodes
        self.theta = theta
        for node in self.nodes:
            node.edges.append(self)
        self.name = "("+",".join([node.name for node in self.nodes])+")"

    def __str__(self):
        return self.name

class FactorNode(object):
    '''
    a factor node class for use in factor models, basically makes running sum-product algorithm easier to think about.

    adjacent: list of nodes that contribute to this factor.
    pending: list of nodes that need to be sent a new message.
              a node N is pending if we've received a message from another node since we last sent a message to N.
    incoming_messages: messages that we've received.
    func: actual function in this factor
    name: name for this factor. Looks like (node1,node2,node3,...) for nodes in thie factor.
    '''
    def __init__(self,adjacent,func):
        self.adjacent = adjacent
        self.pending = set(adjacent)
        for node in adjacent:
            node.adjacent.append(self)
            node.pending.add(self)
        self.incoming_messages = {}
        self.func = func
        self.name = "("+",".join([node.name for node in self.adjacent])+")"

    def __str__(self):
        return self.name

    def getmessage(self,node,message,debug):
        if(debug):
            print "factor "+str(self)+" got message from "+str(node)
        self.incoming_messages[node] = message
        addpending = set(self.adjacent)
        addpending.remove(node)
        self.pending = addpending.union(self.pending)
    
    def sendmessages(self,debug=False):
        if(len(self.pending)==0):
            return False
        for node in self.pending:
            message = self.computemessage(node,debug)
            node.getmessage(self,message,debug)
        self.pending = set([])
        return True

    def computemessage(self,node,debug):
        
        message = {}

        values = np.zeros(len(self.adjacent))
        if(debug):
            print "computing message from factor "+str(self)+" to "+str(node)+" with known messages:"
            for m in self.incoming_messages:
                print str(m)+": " +str(self.incoming_messages[m])
        for x in node.observedsupport:
            message[x] = self.recursivesum(0,node,x,values,1,0,debug)

        if(debug):
            print "message from factor "+str(self)+" to node "+str(node)+":"
            print "\t"+str(message)
        return message

    def recursivesum(self,index,node,x,values,passedmessage,total,debug):
        if(index == len(self.adjacent)):
            fs = self.func(values)

            if(debug):
                print "\t value: "+str(values)+" messageproduct: ",passedmessage
            return fs*passedmessage

        if(self.adjacent[index]==node):
            values[index] = x
            return self.recursivesum(index+1,node,x,values,passedmessage,total,debug)

        support = self.adjacent[index].observedsupport
        for xm in support:
            values[index] = xm
            try:
                newpassedmessage = passedmessage*self.incoming_messages[self.adjacent[index]][xm]
            except KeyError:
                newpassedmessage = passedmessage
            total += self.recursivesum(index+1,node,x,values,newpassedmessage,total,debug)
        return total

    
class GraphicalModel(object):
    '''
    implements operations on graphical models. currently only really does Ising models though.
    
    nodes: list of variables in the model
    edges: list of edges
    factors: factors for use in constructing a factor graph
    havepending: a set of nodes and factors that have pending messages to send.
    '''
    def __init__(self,nodes,edges):
        self.nodes = nodes
        self.edges = edges
        self.factors = []
        self.havepending = set([])

        def fmaker(theta):
            return lambda values:np.exp(theta*reduce(lambda a,b:a*b,values))
        for node in self.nodes:
            f = fmaker(node.theta)
            self.factors.append(FactorNode([node],f))

        for edge in self.edges:
            f = fmaker(edge.theta)
            self.factors.append(FactorNode(edge.nodes,f))
        self.havepending = set(self.nodes+self.factors)


    def getbyname(self,name):
        nodes = filter(lambda n:n.name==name,self.nodes)
        if(len(nodes)==0):
            return None
        return nodes[0]

    def observenode(self,name,value):
        self.getbyname(name).observedsupport = [value]

    def observenodes(self,vals):
        for name in vals:
            self.observenode(name,vals[name])

    def resetallnodes(self):
        for node in self.nodes:
            node.observedsupport = deepcopy(node.support)


    def printmessages(self):
        print "messages in this model:"
        print "Messages TO Factors: "
        for factor in self.factors:
            print "\tfrom: "+str(factor)
            for m in factor.incoming_messages:
                print "\t\t" +str(m)+": " +str(factor.incoming_messages[m])
        print "Messages TO Variables: "
        for node in self.nodes:
            print "\tfrom: "+str(node)
            for m in node.incoming_messages:
                print "\t\t" +str(m)+": " +str(node.incoming_messages[m])

    def propogate(self,depth = 100,debug = False):
        d = 0
        dmax = depth*len(self.havepending)
        while(d<dmax and len(self.havepending)!=0):
            sender = self.havepending.pop()
            if(sender.sendmessages(debug)):
                self.havepending = self.havepending.union(set(sender.adjacent))
            d+=1

        return d<dmax
        
