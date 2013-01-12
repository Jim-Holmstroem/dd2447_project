import random
import numpy as np
import itertools as it

class queen_generator:
    def __init__(self, N, dim = 2):
        self.N = N
        self.dim = dim
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
        for solution in self.solutions(self.empty_board()): #Just passing on the yield
            yield solution

    def solutions(self, state):
        """docstring for solutions
        
        yields solutions 
        """
        for new_state in queen_generator.valid_states( state ):
            num_queens = np.sum(np.sum(new_state))
            if( num_queens < self.N ):
                self.solutions(new_state)
            elif( num_queens == self.N ):
                yield new_state
            else:
                raise Exception(
                    "Too many queens on the field, \
                    something is wrong with the algorithm"
                    )

    @staticmethod
    def valid_states( state ):
        """docstring for valid_coordinates"""
        pass 

    @staticmethod
    def iterate_empty_coordinates( state ):
        """docstring for iterate_empty_coordinates

        iterates tuple (x,y) for all coordintes which are free
        """
        coords = map(lambda axis:
                    np.nonzero(state.any(axis=axis))[0],
                    range(dim)
                )
    
        return it.product(*coords)

    def empty_board(self):
        return np.zeros((self.N,self.N),dtype=bool)

for s in queen_generator(5):
    print s

