#!/usr/bin/python

import operator
import random

observation_missrate = 0.05

class Graph(object):
    O, L, R = '0', 'L', 'R'
    
    def __init__(self, matrix):
        self.matrix = matrix

    def adjacent_connections(self, v_from):
        # Obtain other connections
        return [ (index, connection) for (index, connection)
                        in enumerate(self.matrix[v_from]) if connection != 0 ]

    def c(self, v_from, e, t, o, p = observation_missrate):
        v_to = e[0]
        edge = e[1]
        
        
        if t == 0:
            return 1.0 / len(self.matrix)
        
        if e[1] not in self.matrix[v_from][v_to]:
            raise IndexError()
        
        connections = self.adjacent_connections( v_from )
        other_connections = [ (index, tp) for (index, connection) in connections
                              for tp in connection if (index, tp) != e ]
        
        print v_from, "->", v_to, "along", edge, "observation", o[t]
        
        
        probability = ( p, (1 - p) ) [o[t] == edge]
        
        if edge == Graph.O:
            return probability * sum([ self.c(v_prev, (v_from, edge_prev), t - 1, o) 
                                       for (v_prev, edge_prev) in other_connections ])
        else:
            f = [ (v_from, edge_prev) for (v_from, edge_prev) in other_connections if edge_prev == Graph.O ][0]
            
            return probability * self.c(f[0], (v_from, f[1]), t - 1, o)

# p(s, O|sigma,G)
def state_probability(graph, v_from, e, o, sigma):
    return reduce(operator.mul, [ graph.c(v_from, e, i, o) for i in range(len(o)) ] )

# p(sigma|O,G)
def sigma_probability(sigma, o, graph, v_from, e):
    return state_probability(graph, v_from, e, o, sigma)

# p(sigma|O,G)
def MCMC(graph, o, sampling_burn_in, sampling_interval, number_of_samples)
    """docstring for MCMC
    returns a sample from the p(sigma|O,G)
    """
    #v_from = random.randint(0, len(graph.matrix) - 1)
    #e = ( random.randint(0, len(graph.matrix) - 1), random.choice([ Graph.L, Graph.R, Graph.O ]) )

    #FIXME: number_of_samples

    for v_from in range(len(graph.matrix)):
        for e in graph.adjacent_connections(v_from):
            # Randomize start sigma
            sigma = random.getrandbits(len(graph.matrix))
            sigma_prob = sigma_probability(sigma, o, graph, v_from, e)

            i = 0

            while True:
                sigma_p = sigma ^ (1 << random.randint(0, len(graph.matrix) - 1))
                sigma_prob_p = sigma_probability(sigma_p, o, graph, v_from, e)

                alpha = sigma_prob_p / sigma_prob

                if alpha > 1 or random.random() <= alpha:
                    sigma, sigma_prob = sigma_p, sigma_prob_p

                    # Sample
                    if i % sampling_interval = 0 and i / sampling_interval > sampling_burn_in:
                        yield sigma_p

                i += 1

# p(O|sigma,G)
def observation_probability(graph,sigma,o):
    """docstring for observation_probability
    
    """
    return sum(
        [graph.c(v_from,e,len(o)-1) 
            for v_from in range(len(graph.matrix)) 
            for e in graph.adjacent_connections(v_from)]
        )

if __name__ == "__main__":
    import sys
    
    # test
    
    matrix = (
        ( 0, ( Graph.L, Graph.R, Graph.O ) ),
        ( ( Graph.L, Graph.R, Graph.O ), 0 )
    )
    
    o = [ Graph.R, Graph.L ]
    
    sigma = ( Graph.R, Graph.R, Graph.L, Graph.L )
    
    graph = Graph(matrix)
    
    # from v = 0, along edge to v = 1 with type 'L'
    print graph.c(0, (1, 'L'), 1, o)
    
    print
    
    # from v = 1, along edge to v = 0 with type 'R'
    print graph.c(1, (0, 'R'), 1, o)
    
    print graph.state_probability(0, (1, 'L'), o, sigma)
