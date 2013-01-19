#!/usr/bin/python

import operator
import random
import sys

observation_missrate = 0.05
number_of_samples = 100
sampling_burn_in = 500
sampling_interval = 5

class Graph(object):
    O, L, R = '0', 'L', 'R'
    
    def __init__(self, matrix):
        self.matrix = matrix

    def adjacent_connections(self, v_from):
        # Obtain other connections
        connections = [ (index, connection) for (index, connection)
                        in enumerate(self.matrix[v_from]) if connection != 0 ]
        
        return [ (index, tp) for (index, connection) in connections
                              for tp in connection ]

    def c(self, v_from, e, t, o, p = observation_missrate):
        v_to = e[0]
        edge = e[1]
        
        
        if t == 0:
            return 1.0 / len(self.matrix)
        
        if edge not in self.matrix[v_from][v_to]:
            print "v_from:", v_from, "v_to:", v_to, "edge:", edge
            raise IndexError()
        
        connections = self.adjacent_connections( v_from )
        other_connections = [ (index, tp) for (index, connection) in connections
                              for tp in connection if (index, tp) != e ]
        
        #print v_from, "->", v_to, "along", edge, "observation", o[t]
        
        
        probability = ( p, (1 - p) ) [o[t] == edge]
        
        print "v_from:", v_from, "e:", e, "other_conns:", other_connections
        
        if edge == Graph.O:
            return probability * sum([ self.c(v_prev, (v_from, edge_prev), t - 1, o) 
                                        for (v_prev, edge_prev) in other_connections ])
        else:
            f = [ (v_from, edge_prev) for (v_from, edge_prev) in other_connections if edge_prev == Graph.O ][0]
            
            return probability * self.c(f[0], (v_from, f[1]), t - 1, o)

# p(s, O|sigma,G)
def state_observation_probability_given_sigma(graph, v_from, e, o, sigma):
    return reduce(operator.mul, [ graph.c(v_from, e, i, o) for i in range(len(o)) ] )

# p(sigma|O,G)
def sigma_probability(sigma, o, graph, v_from, e):
    return state_observation_probability_given_sigma(graph, v_from, e, o, sigma)

# p(sigma|O,G)
def metropolis_hastings(graph, o, number_of_samples, sampling_burn_in, sampling_interval):
    iterations = (number_of_samples + sampling_burn_in) * sampling_interval / (len(graph.matrix) * 3)
    sigmas = [ 0 for i in range(number_of_samples) ]
    
    burn_in = sampling_burn_in / (len(graph.matrix) * 3)

    for v_from in range(len(graph.matrix)):
        
        for e in graph.adjacent_connections(v_from): #foreach state
            if not valid_state(graph, v_from, e):
                continue
        
            # Randomize start sigma
            sigma = random.getrandbits(len(graph.matrix))
            sigma_prob = sigma_probability(sigma, o, graph, v_from, e)
            
            print "v_from:", v_from, "e:", e
            
            for i in range(iterations):
                sigma_p = sigma ^ (1 << random.randint(0, len(graph.matrix) - 1))
                sigma_prob_p = sigma_probability(sigma_p, o, graph, v_from, e)

                alpha = sigma_prob_p / sigma_prob

                if alpha > 1 or random.random() <= alpha:
                    sigma, sigma_prob = sigma_p, sigma_prob_p

                    # Sample

                    if i % sampling_interval == 0 and i / sampling_interval > burn_in:
                        sigmas = [ sigma_p ]
                        
                        sbin = bin(sigma_p)[2:]
                        print ("0" * (2 - len(sbin))) + sbin
                    
    return sigmas

# p(O|sigma,G)
def observation_probability(graph,sigma,o):
    """docstring for observation_probability
    
    """
    return sum(
        [graph.c(v_from, e, len(o) - 1, o) 
            for v_from in range(len(graph.matrix)) 
            for e in graph.adjacent_connections(v_from)]
        )

'''
def states(graph, sigma):
    all_states = []
    
    for v_from in range(len(graph.matrix)):
        for e in graph.adjacent_connections(v_from): #foreach state
            all_states.append(0)
'''

def valid_state(graph, v_from, e):
    return graph.matrix[v_from][e[0]] != 0

def state_probability(v_from, e, graph, o):
    sigmas = metropolis_hastings(graph, o, number_of_samples, sampling_burn_in, sampling_interval)
    
    return 1.0 / number_of_samples * sum([ 
        state_observation_probability_given_sigma(graph, v_from, e, o, sigma) / 
        observation_probability(graph,sigma,o) for sigma in sigmas 
    ])

if __name__ == "__main__":
    import sys
    
    # test
    
    '''matrix = (
        ( 0, ( Graph.L, Graph.R, Graph.O ) ),
        ( ( Graph.L, Graph.R, Graph.O ), 0 )
    )'''
    
    matrix = (
        ( ( '0', 'R' ), 0, 0, 0, 0, 0, 0, 0, 0, 0, ( 'L' ), 0 ),
        ( 0, 0, 0, ( 'R' ), 0, 0, 0, 0, 0, ( 'L' ), 0, ( '0' ) ),
        ( 0, 0, 0, 0, 0, ( 'L' ), ( '0' ), 0, 0, 0, 0, ( 'R' ) ),
        ( 0, ( 'L' ), 0, 0, 0, 0, ( 'R' ), ( '0' ), 0, 0, 0, 0 ),
        ( 0, 0, 0, 0, 0, ( 'R' ), 0, ( 'L' ), 0, 0, 0, ( '0' ) ),
        ( 0, 0, ( 'L' ), 0, ( 'R' ), 0, 0, ( '0' ), 0, 0, 0, 0 ),
        ( 0, 0, ( '0' ), ( 'R' ), 0, 0, 0, 0, 0, 0, ( 'L' ), 0 ),
        ( 0, 0, 0, ( 'L' ), ( 'R' ), ( '0' ), 0, 0, 0, 0, 0, 0 ),
        ( 0, 0, 0, 0, 0, 0, 0, 0, 0, ( '0', 'L' ), ( 'R' ), 0 ),
        ( 0, ( 'R' ), 0, 0, 0, 0, 0, 0, ( 'L', '0' ), 0, 0, 0 ),
        ( ( 'R' ), 0, 0, 0, 0, 0, ( '0' ), 0, ( 'L' ), 0, 0, 0 ),
        ( 0, ( 'R' ), ( '0' ), 0, ( 'L' ), 0, 0, 0, 0, 0, 0, 0 ),
        )
    
    #o = [ Graph.R, Graph.L ]
    o = [ random.choice([Graph.L, Graph.R, Graph.O]) for i in range(len(matrix)) ]
    
    print "observation array:", o
    
    sigma = ( Graph.R, Graph.R, Graph.L, Graph.L )
    
    graph = Graph(matrix)
    
    # from v = 0, along edge to v = 1 with type 'L'
    #print graph.c(0, (1, 'L'), 1, o)
    
    #print
    
    # from v = 1, along edge to v = 0 with type 'R'
    #print graph.c(1, (0, 'R'), 1, o)
    
    #print state_observation_probability_given_sigma(graph, 0, (1, 'L'), o, sigma)
    
    #print "state_probability:", state_probability(0, (1, 'L'), graph, o)
    
    total_prob = 0.0
    
    for v_from in range(len(graph.matrix)):
        
        for e in graph.adjacent_connections(v_from): #foreach state
            p = state_probability(0, (1, 'L'), graph, o)
            print "state_probability: v_from:", v_from, "e:", e, "p:", p
            total_prob += p

    print "total_prob:", total_prob
    
    
    
