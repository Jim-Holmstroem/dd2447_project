import random
import numpy as np
import itertools as it
from sets import Set
import operator as op

class HashComparatorMixin(object):
    def __lt__(self, other):
        return self.hash()<other.hash()
    def __eq__(self, other):
        return self.hash()==other.hash()
    def __ne__(self, other):
        return self.hash()!=other.hash()
    def __gt__(self, other):
        return self.hash()>other.hash()
    def __ge__(self, other):
        return self.hash()<=other.hash()
    def __le__(self, other):
        return self.hash()>=other.hash()

class BinarySearchTree(object):
    def __init__(self):
        self._top = None

    class Node(HashComparatorMixin):
        def __init__(self, state):
            self._left = None
            self._right = None
            self._key = None
            self._state = state

        def hash(self):
            """docstring for hash

                not a very good one by guaranteed to be unique for 
                don't use __hash__ since it cuts longs to 1 for some reason
            """
            if(self._key is None):
                self._key = sum(map(lambda a:1<<a,self._state.flatten().nonzero()[0]))
                return self._key
            else:
                return self._key

        def append(self,node):
            if(self==node):
                print '=='
                return False
            elif(self<node):
                if(self._left is None):
                    self._left = node
                    print 'left'
                    return True
                else:
                    return self._left.append(node)
            else: #self>node
                if(self._right is None):
                    self._right = node
                    print 'right'
                    return True
                else:
                    return self._right.append(node)

        def __str__(self):
            return "(\n{left},\n{state}({key}),\n{right}\n)".format(
                        left=self._left,
                        right=self._right,
                        key=self._key,
                        state=self._state
                    )

    def exists(self,state):
        """docstring for append"""
        node = BinarySearchTree.Node(state)
        if(self._top is None):
            self._top = node
        else:
            self._top.append(node)

    def __str__(self):
        return str(self._top)

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

