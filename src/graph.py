#!/usr/bin/python

import operator
import random
import sys
import numpy as np
import pylab as pl
import math
import multiprocessing as mp


observation_missrate = 0.05
number_of_samples = 2000
sampling_burn_in = 200
sampling_interval = 5

# Utility function for printing sigmas
def print_sigma(sigma):
    sbin = bin(sigma)[2:]
    print ("0" * (16 - len(sbin))) + sbin

class StateSpace(object):
    def __init__(self, graph):
        self.graph = graph
    
    def __iter__(self):
        for v_from in range(len(self.graph.matrix)):
            for e in graph.adjacent_connections(v_from): #foreach state
                if valid_state(self.graph, v_from, e):
                    yield (v_from, e)

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
    
    def opposite_connection(self, v_from, e):
        connections = [ (index, connection) for (index, connection)
                        in enumerate(self.matrix[v_from]) if connection != 0 ]
        for connection in connections:
            for edge in connection[1]:
                if edge == e:
                    return (connection[0], self.invert_connection_direction(v_from, connection[0], edge))
    
    def connections_to(self, v_to):
        
        connections = [ (from_index, e) for (from_index, row) in enumerate(self.matrix) for (to_index, e) in enumerate(row) if to_index == v_to and e != 0 ]
        
        #print "connections_to:", connections
        
        return [ (index, tp) for (index, connection) in connections
                 for tp in connection ]

    def invert_connection_direction(self, v_from, v_to, edge):
        return self.matrix[v_to][v_from][self.matrix[v_from][v_to].index(edge)]
    
    def target(self, v_from, edge):
        return [ index for (index, connection) in enumerate(self.matrix[v_from]) if connection != 0 and edge in connection ][0]

class Observation(object):
    def __init__(self, graph, o):
        self.graph = graph
        self.o = o
        
        self.c_cache = np.ones((len(graph.matrix), 3, len(o)))*np.NaN
    
    def __label_to_index(self, l):
        if l == Graph.L:
            return 0
        elif l == Graph.R:
            return 1
        elif l == Graph.O:
            return 2
        raise IndexError()
    
    def c(self, v_from, e, t, p = observation_missrate):
        edge = self.__label_to_index(e[1])
        
        c = self.c_cache[v_from, edge, t]
        
        if np.isnan(c):
            c = self.__c(v_from, e, t, p)
            self.c_cache[v_from, edge, t] = c
        
        return c
    
    def __c(self, v_from, e, t, p = observation_missrate):
        v_to = e[0]
        edge = e[1]
        
        backward_edge = self.graph.invert_connection_direction(v_from, v_to, edge)
        
        if t == 0:
            return 1.0 / len(self.graph.matrix)
        
        if edge not in self.graph.matrix[v_from][v_to]:
            print "v_from:", v_from, "v_to:", v_to, "edge:", edge
            raise IndexError()

        connections = self.graph.connections_to(v_from)
        
        other_connections = [ (index, tp) for (index, connection) in connections
                              for tp in connection if (index, tp) != (v_to, backward_edge) ]
        
        
        probability = ( p, (1 - p) ) [self.o[t] == edge]

        if edge == Graph.O:
            return probability * sum([ self.c(v_prev, (v_from, edge_prev), t - 1) 
                                        for (v_prev, edge_prev) in other_connections ])
        else:
            f = [ (v, edge_prev) for (v, edge_prev) in other_connections if self.graph.invert_connection_direction(v, v_from, edge_prev) == Graph.O ][0]
            
            return probability * self.c(f[0], (v_from, f[1]), t - 1)

    def d(self, v_from, e, sigma, p = observation_missrate):
        edge = e[1]
        probability = 1.0
        
        for o in reversed(self.o):
            if edge == Graph.O:
                # Choose either L or R using sigma
                backward_edge = [Graph.L, Graph.R][sigma_get_bit(sigma, v_from)]
            else:
                # Choose 0
                backward_edge = Graph.O

            probability *= [p / 2, 1 - p][o == edge]
            
            v_to = self.graph.target(v_from, backward_edge)
            edge = self.graph.invert_connection_direction(v_from, v_to, backward_edge)
            v_from = v_to
        
        return probability


def generate_random_observation_sequence(graph, length):
    # end state
    e_stop = random.choice([Graph.L, Graph.R, Graph.O])
    v_stop = random.randint(0, len(graph.matrix)),
    
    return Observation(graph, [ random.choice([Graph.L, Graph.R, Graph.O]) for i in range(length) ]), \
                       v_stop, (graph.target(v_stop, e), e_stop)

def generate_observation_sequence(graph, length, p = observation_missrate):
    v_from = random.randint(0, len(graph.matrix) - 1)
    e_from = random.choice([Graph.L, Graph.R, Graph.O])
    o = [0] * length
    
    def next_state(v_from, e_from):
        v_to, e_to = graph.opposite_connection(v_from, e_from)
        
        if e_to == Graph.O:
            return v_to, random.choice([Graph.L, Graph.R])
        else:
            return v_to, Graph.O
    
    for i in range(length):
        v_from, o[i] = next_state(v_from, e_from)
    
    return Observation(graph, o), v_from, ( graph.target(v_from, o[length - 1]), o[length - 1])

def generate_noisy_observation_sequence(graph, length, p = observation_missrate):
    observation, v_from, e = generate_observation_sequence(graph, length, p)
    labels = ( Graph.L, Graph.R, Graph.O )
    
    for i in range(len(observation.o)):
        if random.random() <= p:
            observation.o[i] = random.choice([ l for l in labels if l != observation.o[i] ])
    
    return observation, v_from, e

# p(s, O|sigma,G)
def state_observation_probability_given_sigma(observation, v_from, e, sigma):
    #print "state_observation_probability_given_sigma", v_from, e, o, sigma
    
    #p = reduce(operator.mul, [ observation.c(v_from, e, i) for i in range(len(observation.o)) ] )
    
    #return p
    
    #return observation.c(v_from, e, len(observation.o) - 1)
    return observation.d(v_from, e, sigma)

# sample measurement for p(sigma|O,G)
def proportional_sigma_probability(sigma, observation, v_from, e):
    return state_observation_probability_given_sigma(observation, v_from, e, sigma)

def random_sigma(graph):
    return random.getrandbits(len(graph.matrix))

def sigma_flip_random_bit(sigma):
    return sigma ^ (1 << random.randint(0, len(observation.graph.matrix) - 1))

def sigma_get_bit(sigma, index):
    return bool(sigma & (1 << index))

# raw metropolis_hastings takes a state
def raw_metropolis_hastings(observation, v_from, e, number_of_samples):
    sigmas = np.empty(number_of_samples, dtype = object)
    probabilities = np.empty(number_of_samples, dtype = float)
    
    # Randomize start sigma
    sigma = random_sigma(observation.graph)
    
    sigma_prob = proportional_sigma_probability(sigma, observation, v_from, e)
    
    for i in range(number_of_samples):
        while True:
            sigma_p = sigma_flip_random_bit(sigma)
            
            sigma_prob_p = proportional_sigma_probability(sigma_p, observation, v_from, e)

            alpha = sigma_prob_p / sigma_prob
            
            probabilities[i] = sigma_prob_p

            if alpha >= 1 or random.random() <= alpha:
                sigma, sigma_prob = sigma_p, sigma_prob_p
                
                sigmas[i] = sigma
                probabilities[i] = sigma_prob
                
                break

    return sigmas, probabilities

# mcmc sampling of p(sigma|O,G)
def sampler(f, observation, number_of_samples, sampling_burn_in, sampling_interval):
    concat_sigmas = []
    concat_probabilities = []
    
    for (v_from, e) in StateSpace(observation.graph):
        sigmas, probabilities = f(observation, v_from, e, (number_of_samples + sampling_burn_in) * sampling_interval / (len(observation.graph.matrix) * 3))
        concat_sigmas.extend(sigmas[sampling_burn_in * sampling_interval / (len(observation.graph.matrix) * 3)::sampling_interval])
        concat_probabilities.extend(probabilities[sampling_burn_in * sampling_interval/ (len(observation.graph.matrix) * 3)::sampling_interval])
    
    return concat_sigmas, concat_probabilities

# mcmc sampling of p(sigma|O,G)
def metropolis_hastings(observation, number_of_samples, sampling_burn_in, sampling_interval):
    iterations = (number_of_samples + sampling_burn_in) * sampling_interval / (len(observation.graph.matrix) * 3)
    sigmas = [ ]
    probabilities = [ ]
    
    burn_in = sampling_burn_in / (len(observation.graph.matrix) * 3)

    for (v_from, e) in StateSpace(observation.graph):
    
        # Randomize start sigma
        sigma = random.getrandbits(len(observation.graph.matrix))
        
        sigma_prob = proportional_sigma_probability(sigma, observation, v_from, e)
        
        #print "metropolis_hastings", "v_from:", v_from, "e:", e
        
        for i in range(iterations):
            sigma_p = sigma ^ (1 << random.randint(0, len(observation.graph.matrix) - 1))
            sigma_prob_p = proportional_sigma_probability(sigma_p, observation, v_from, e)

            alpha = sigma_prob_p / sigma_prob
            
            probabilities.append(sigma_prob_p)

            if alpha > 1 or random.random() <= alpha:
                sigma, sigma_prob = sigma_p, sigma_prob_p

                # Sample

                if i % sampling_interval == 0 and i / sampling_interval > burn_in:
                    sigmas.append(sigma_p)

    return ( sigmas, probabilities )

# p(O|sigma,G)
def observation_probability(observation, sigma):
    """docstring for observation_probability
    
    """
    
    return sum(
        [observation.d(v_from, e, sigma) 
            for (v_from, e) in StateSpace(observation.graph) ]
        )
    
    '''return sum(
        [observation.c(v_from, e, len(observation.o) - 1) 
            for v_from in range(len(graph.matrix)) 
            for e in observation.graph.adjacent_connections(v_from)]
        )'''

def valid_state(graph, v_from, e):
    return observation.graph.matrix[v_from][e[0]] != 0

def state_probability(f, v_from, e, observation):
    sigmas, _ = sampler(f, observation, number_of_samples, sampling_burn_in, sampling_interval)
    
    return 1.0 / number_of_samples * sum([ 
        state_observation_probability_given_sigma(observation, v_from, e, sigma) / 
        observation_probability(observation,sigma) for sigma in sigmas 
    ])

def trace_plot(f, observation, filename):
    length = (number_of_samples + sampling_burn_in) * sampling_interval / (len(observation.graph.matrix) * 3)
    
    for (v_from, e) in StateSpace(observation.graph):
        
        p = np.zeros(length, dtype = float)
        
        for i in range(50):
            _, probabilities = f(observation, v_from, e, length)
            
            for j in range(length):
                p[j] += probabilities[j]
        
        pl.semilogy([ prob / length for prob in p ])
        #break
        
    pl.xlabel('interval')
    pl.ylabel('log(probability) + k')
    pl.title('Convergence plot')
    pl.grid(True)
    pl.savefig(filename, bbox_inches = 0)
    
    pl.show()
    
if __name__ == "__main__":
    import sys
    
    # test
    
    '''matrix = (
        ( 0, ( Graph.L, Graph.R, Graph.O ) ),
        ( ( Graph.L, Graph.R, Graph.O ), 0 )
    )'''
    
    #random.seed(3453)
    
    matrix = (
        ( ( '0', 'R' ), 0, 0, 0, 0, 0, 0, 0, 0, 0, ( 'L' ), 0 ),    # 0
        ( 0, 0, 0, ( 'R' ), 0, 0, 0, 0, 0, ( 'L' ), 0, ( '0' ) ),   # 1
        ( 0, 0, 0, 0, 0, ( 'L' ), ( '0' ), 0, 0, 0, 0, ( 'R' ) ),   # 2
        ( 0, ( 'L' ), 0, 0, 0, 0, ( 'R' ), ( '0' ), 0, 0, 0, 0 ),   # 3
        ( 0, 0, 0, 0, 0, ( 'R' ), 0, ( 'L' ), 0, 0, 0, ( '0' ) ),   # 4
        ( 0, 0, ( 'L' ), 0, ( 'R' ), 0, 0, ( '0' ), 0, 0, 0, 0 ),   # 5
        ( 0, 0, ( '0' ), ( 'R' ), 0, 0, 0, 0, 0, 0, ( 'L' ), 0 ),   # 6
        ( 0, 0, 0, ( 'L' ), ( 'R' ), ( '0' ), 0, 0, 0, 0, 0, 0 ),   # 7
        ( 0, 0, 0, 0, 0, 0, 0, 0, 0, ( '0', 'L' ), ( 'R' ), 0 ),    # 8
        ( 0, ( 'R' ), 0, 0, 0, 0, 0, 0, ( 'L', '0' ), 0, 0, 0 ),    # 9
        ( ( 'R' ), 0, 0, 0, 0, 0, ( '0' ), 0, ( 'L' ), 0, 0, 0 ),   # 10
        ( 0, ( 'R' ), ( '0' ), 0, ( 'L' ), 0, 0, 0, 0, 0, 0, 0 ),   # 11
        )
    '''matrix = (
        ( 0, ( 'R', '0' ), ( 'L', ), 0 ),
        ( ( 'L', '0' ), 0, 0, ( 'R', ) ),
        ( ( '0', ), 0, 0, ( 'R', 'L' ) ),
        ( 0, ( 'L', ), ( '0', 'R' ), 0 ),
    )'''
    
    observation_sequence_length = 14
    
    #o = [ Graph.R, Graph.L ]
    #o = [ random.choice([Graph.L, Graph.R, Graph.O]) for i in range(len(matrix)) ]
    
    graph = Graph(matrix)
    
    observation, v_from, e = generate_observation_sequence(graph, observation_sequence_length)
    
    print "observation array:", observation.o, "v_from:", v_from, "e:", e
    
    #sigma = ( Graph.R, Graph.R, Graph.L, Graph.L )
    
    #observation = Observation(graph, o)
    
    # from v = 0, along edge to v = 1 with type 'L'
    #print graph.c(0, (1, 'L'), 1, o)
    
    #print
    
    # from v = 1, along edge to v = 0 with type 'R'
    #print graph.c(1, (0, 'R'), 1, o)
    
    #print state_observation_probability_given_sigma(graph, 0, (1, 'L'), o, sigma)
    
    #print "state_probability:", state_probability(0, (1, 'L'), graph, o)
    
    #total_prob = 0.0
    
    '''for v_from in range(len(graph.matrix)):
        
        for e in graph.adjacent_connections(v_from): #foreach state
            p = state_probability(0, (10, 'L'), observation)
            #print "state_probability: v_from:", v_from, "e:", e, "p:", p
            total_prob += p'''
    
    probabilities = []
    states = [ (v_from, e) for (v_from, e) in StateSpace(observation.graph) ]
    
    for v_from, e in states:
        p = state_probability(raw_metropolis_hastings, v_from, e, observation)
        #total_prob += p
        
        probabilities.append(p)
        
        print "state_probability(", v_from, e, ") =", p

    max_index = probabilities.index(max(probabilities))
    
    print "state:", states[max_index], "with probability:", probabilities[max_index]

    #print "total_prob:", total_prob
    
    print "state_probability:", state_probability(raw_metropolis_hastings, v_from, e, observation)
    
    print "total_prob:", sum(probabilities)
    
    trace_plot(raw_metropolis_hastings, observation, '../plot/state_' + str(states[max_index][0]) + "_" + str(states[max_index][1][0]) + "_" + str(states[max_index][1][1]) + "_probability_" + str(probabilities[max_index]) + "_total_prob_" + str(sum(probabilities)) + "_observation_length_" + str(len(observation.o)) + ".pdf")
    
    # test
    
    #print "state_observation_probability_given_sigma:", state_observation_probability_given_sigma(Observation(graph, [ 'R', '0', 'R', '0' ]), 1, (0, Graph.O), 0)
    
