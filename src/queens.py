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
        def clash(state):
            """docstring for clash"""
            return False
        self.constraints.append(clash):
        def no_self_junction_connection(state, m):
            """docstring for no_self_junction_connection
                
            make sures that we  
            """
            
            pass
     
    def __iter__(self):
        
        
    

for s in queen_generator(5):
    print s

