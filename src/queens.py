import random
import numpy as np


class queen_generator:
    def __init__(self,N):
        self.N = N
        self.constraints = [] 
        def no_self_connection(state):
            """docstring for no_self_connection"""
            return np.any(np.diag(state))
        self.constraints.append(no_self_connection)
        def no_self_junction_connection(state, m):
            """docstring for no_self_junction_connection
                
            make sures that we don't have connections in a junction of size `m`
            that is no ones in the boxes of size m by m along the diagonal
            """
            return False        
        self.constraints.append(no_self_junction_connection)

    def __iter__(self):
        for solution in self.solutions(): #Just passing on the yield
            yield solution

    def solutions(self,state=None):
        """docstring for solutions"""
        if(state is None):
            state = np.zeros((self.N,self.N))
        if(np.sum(np.sum(state))==self.N):
            yield state
        else
            

for s in queen_generator(5):
    print s

