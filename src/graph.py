#!/usr/bin/python

import operator

observation_missrate = 0.05

class Graph(object):
    O, L, R = '0', 'L', 'R'
    
    def __init__(self, matrix):
        self.matrix = matrix

    def c(self, v_from, e, t, o, p = observation_missrate):
        v_to = e[0]
        edge = e[1]
        
        
        if t == 0:
            return 1.0 / len(self.matrix)
        
        if e[1] not in self.matrix[v_from][v_to]:
            raise IndexError()
        
        # Obtain other connections
        connections = [ (index, connection) for (index, connection)
                        in enumerate(self.matrix[v_from]) if connection != 0 ]

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
    
    # p(s, O|G, sigma)
    def state_probability(self, v_from, e, o, sigma):
        return reduce(operator.mul, [ self.c(v_from, e, i, o) for i in range(len(o)) ] )

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
