import random
import numpy as np
import itertools as it

class queen_generator:
    """docstring for queen generator

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
        def no_self_connection(state):
            """docstring for no_self_connection"""
            return not np.any(np.diag(state))
        self.constraints.append(no_self_connection)
        
        # SHOULD BE APPENDED EXTERNALLY LATER
        #def no_self_junction_connection(state, m):
        #    """docstring for no_self_junction_connection
        #        
        #    make sures that we don't have connections in a junction of size `m`
        #    that is no ones in the boxes of size m by m along the diagonal
        #    """
        #    return False        
        #self.constraints.append(no_self_junction_connection)

    def __iter__(self):
        for solution in self.solutions(self.empty_board()): #Just passing on the yield
            yield solution

    def solutions(self, state):
        """docstring for solutions
        
        yields solutions 
        """
        for new_state in self.valid_states( state ):
            print new_state
            num_queens = np.sum(np.sum(new_state))
            if( num_queens < self.N ):
                print list(self.solutions(new_state))
            elif( num_queens == self.N ):
                yield new_state
            else:
                raise Exception(
                    "Too many queens on the field, \
                    something is wrong with the algorithm"
                    )

    def valid_states( self, state ):
        """docstring for valid_coordinates"""
        for possible_coordinate in queen_generator.iterate_empty_coordinates( state ):
            if(state[possible_coordinate]==True):
                raise Exception(
                        "Already filled at {coord}, \
                        something went wrong".format(
                            coord = possible_coordinate
                        )
                    )
            new_state = state.copy()
            new_state[possible_coordinate] = True
            #print "new_state=\n{state}".format(state=new_state)
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



#b = queen_generator(5).empty_board()
#b[0,1] = 1
#b[2,0] = 1
#b[3,2] = 1
#print b

#for t in queen_generator.iterate_empty_coordinates(b):
#    print t,b[t]

for s in queen_generator(3):
    print ss

