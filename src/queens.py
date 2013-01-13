import random
import numpy as np
import itertools as it
from sets import Set
import operator as op

class HashComparatorMixin(object):
    def __lt__(self, other):
        return self.hash()<other.hash()
    def __eq__(self, other):
        return self.hash()==self.hash()
    def __ne__(self, other):
        return self.hash()!=self.hash()
    def __gt__(self, other):
        return other.hash()<self.hash()
    def __ge__(self, other):
        return self.hash()<=other.hash()
    def __le__(self, other):
        return other.hash()>=self.hash()

class BinarySearchTree(object):
    def __init__(self):
        self.top = None

    class Node(HashComparatorMixin):
        def __init__(self,state):
            self.left = None
            self.right = None
            self.state = state
            self.key = None 

        def hash(self):
            """docstring for hash

                not a very good one by guaranteed to be unique for 
                don't use __hash__ since it cuts longs to 1 for some reason
            """
            if(self.key):
                return self.key
            else:
                self.key = sum(map(lambda a:1<<a,self.state.flatten().nonzero()[0]))
                print self.key
                return self.key

        def append(self,state):
            node=BinarySearchTree.Node(state)
            if(self==node):
                return False
            elif(self<node):
                if(self.left is None):
                    self.left = node
                    return True
                else:
                    return self.left.append(node)
            else: #self>node
                if(self.right is None):
                    self.right = node
                    return True
                else:
                    return self.right.append(node)
        
        def __str__(self):
            return "({left},{state},{right})".format(
                        left=self.left,
                        right=self.right,
                        state=self.state
                    )


    def exists(self,state):
        """docstring for append"""
        if(self.top is None):
            self.top = BinarySearchTree.Node(state)
        else:
            self.top.append(state)

    def __str__(self):
        return str(self.top)

class ReducedQueenGenerator(object):
    """docstring for queen generator

    without constraints it produces o solution with rooks instead of queens.

    qg = queengenerator(SIZE)
    qg.constraints.append(ADDITIONAL_CONDITION)
    for t in qg:
        print t

    where SIZE is the size of the board and ADDITIONAL_CONDITION is a 
    function c:matrix->bool (for example no connections within pairs or sum of all sub-diagonals must be less then 3)
    """

    def __init__(self, N):
        self.N = N
        self.constraints = [] 
        self.states = []
        self.searchtree = BinarySearchTree()

    @staticmethod
    def no_self_connection(state):
        """docstring for no_self_connection"""
        return not np.any(np.diag(state))

    @staticmethod
    def queens(state):
        """docstring for queens"""
        return

    def __iter__(self):
        start = self.empty_board()
        self.states.append(start)
        self.searchtree.exists(start)
        while(self.states):
            state = self.states.pop()
            num_queens = np.sum(np.sum(state))
            if( num_queens < self.N ):
                for valid_state in self.valid_states(state):
                    if(not self.searchtree.exists(valid_state)):
                        self.states.append( valid_state )
                    else:
                        print "computer says no.."
            elif( num_queens == self.N ): #GOAL, N queens without conflict
                yield state
            else:
                raise Exception(
                    "Too many queens on the field, \
                    something is wrong with the algorithm"
                    )

    def valid_states(self,state):
        """docstring for valid_coordinates"""
        for possible_coordinate in ReducedQueenGenerator.iterate_empty_coordinates( state ):
            if(state[possible_coordinate]):
                raise Exception(
                        "Already filled at {coord}, \
                        something went wrong".format(
                            coord = possible_coordinate
                        )
                    )
            new_state = state.copy()
            new_state[possible_coordinate] = True
            if( all(map(lambda c: c(new_state), self.constraints)) ):
                yield new_state

    @staticmethod
    def iterate_empty_coordinates( state ):
        """docstring for iterate_empty_coordinates

        iterates tuple (x,y) for all coordintes which are free
        """
        coords = map(lambda axis:
                    np.nonzero(state.any(axis=axis)==False)[0], #==False is the easiest way todo elementwise `not`
                    reversed(range(2))
                )
   
        return it.product(*coords)

    def empty_board(self):
        return np.zeros(
                (self.N, self.N),
                dtype=bool
            )

if __name__ == "__main__":

    #b = ReducedQueenGenerator(5).empty_board()
    #b[0,1] = 1
    #b[2,0] = 1
    #b[3,2] = 1
    #print b

    #for t in queen_generator.iterate_empty_coordinates(b):
    #    print t,b[t]
    rq = ReducedQueenGenerator(3)
    rq.constraints.append(ReducedQueenGenerator.no_self_connection)
    
    for s in rq:
        print s
   
    print rq.searchtree
