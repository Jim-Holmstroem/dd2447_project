#include <iostream>
#include <iterator>
#include <functional>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <sys/time.h>
#include <fstream>

/*
 * mcmc metropolis_hastings where
 * ProbabilityFunctor : std::unary_function<VariableT, ProbabilityT>
 * SamplerFunctorT :: std::unary_function<VariableT, void>
 * 
 */

template<typename VariableT, typename QFunctorT, typename ProbabilityFunctorT, typename SamplerFunctorT>
void metropolis_hastings(VariableT x, const QFunctorT & Q,
                         ProbabilityFunctorT & p,
                         SamplerFunctorT & sampler, size_t num_samples) {
    
    typename ProbabilityFunctorT::result_type x_p(p(x));
    
    for(size_t i = 0; i < num_samples; ++i) {
        for(;;) {
            VariableT x_prim(Q(x));
            typename ProbabilityFunctorT::result_type x_prim_p(p(x_prim)),
                                             a = x_prim_p / x_p;
            
            //std::cerr << "a: " << a << std::endl;
            
            if (a >= typename ProbabilityFunctorT::result_type(1) ||
                typename ProbabilityFunctorT::result_type(rand()) / RAND_MAX <= a) {
                
                x = x_prim;
                x_p = x_prim_p;
                
                // TODO:
                sampler(x, x_p);
                
                break;
            }
        }
    }
}

/*
 * sampler to sequence
 */

template<typename VariableT, typename ProbabilityT, typename IteratorT>
class sampler : public std::binary_function<VariableT, ProbabilityT, void> {
    public:
        inline sampler(IteratorT & it,
                       size_t burn_in, size_t sampling_interval) : m_it(it), 
                                                                   m_i(0),
                                                                   m_burn_in(burn_in),
                                                                   m_sampling_interval(sampling_interval) {
            
        }
        
        inline void operator() (VariableT x, ProbabilityT probability) {
            if (m_i % m_sampling_interval == 0 && m_i >= m_burn_in) {
                m_it = std::make_pair(x, probability);
                ++m_it;
            }
            
            ++m_i;
        }
    
    private:
        IteratorT & m_it;
        size_t m_i, m_burn_in, m_sampling_interval;
};

/*
 *
 */

typedef unsigned int sigma_t;
typedef double probability_t;
//typedef unsigned int state_t;
typedef char edge_t;

struct state_t {
    unsigned int v;
    edge_t e;
    
    inline state_t(unsigned int v_ = 0, edge_t e_ = 0) : v(v_), e(e_) { }
    
    inline void set(unsigned int v_, edge_t e_) {
        v = v_;
        e = e_;
    }
};

std::ostream & operator<<(std::ostream & stream, const state_t & state) {
    stream << "v: " << state.v << ", e:" << state.e;
    
    return stream;
}

const edge_t GRAPH_L = 'L', GRAPH_R = 'R', GRAPH_0 = '0';
const edge_t edges[] = { GRAPH_L, GRAPH_R, GRAPH_0 };
const size_t num_edges = 3;
probability_t p = 0.05;

/*
 * Q function emits neighbours to sigma
 */

template<size_t NodeCount>
class Q : public std::unary_function<sigma_t, sigma_t> {
    public:
        inline sigma_t operator() (sigma_t sigma) const {
            return sigma ^ (1 << (rand() % NodeCount));
        }
};

/*
 * switch setting from sigma
 */

edge_t switch_setting(sigma_t sigma, const state_t & state) {
    return (sigma & (1 << state.v)) ? GRAPH_R : GRAPH_L;
}

/*
 * generate random sigma
 */

template<size_t NodeCount>
sigma_t random_sigma() {
    sigma_t sigma(0);
    
    for(size_t i = 0; i < NodeCount; ++i)
        sigma ^= (rand() % 2) << i;
    
    return sigma;
}

/*
 * graph class
 */

template<size_t NodeCount>
class graph_t {
    public:
        typedef char connection_type_t;
        //typedef std::vector<connection_type_t> connection_t;
        typedef std::string connection_t;
 
        const connection_t *operator[] (size_t index) const {
            return m_graph[index];
        }
        
        state_t previous(const state_t & state, sigma_t sigma) const;
        state_t next(const state_t & state, sigma_t sigma) const;
        
        /*
         * parse constant expression
         */
        
        void parse(connection_t graph[NodeCount][NodeCount]) {
            for(size_t y = 0; y < NodeCount; ++y) {
                for(size_t x = 0; x < NodeCount; ++x)
                    m_graph[y][x] = graph[y][x];
            }
        }
        
    private:
        connection_t m_graph [NodeCount][NodeCount];
};

/*
 * functor for finding vertex containing edge
 */

template<size_t NodeCount>
class vertex_contains_edge : public std::unary_function<typename graph_t<NodeCount>::connection_t, bool> {
    public:
        inline vertex_contains_edge(edge_t edge) : m_edge(edge) {
            
        }
        
        inline bool operator() (const typename graph_t<NodeCount>::connection_t & conn) const {
            return conn.find(m_edge) != graph_t<NodeCount>::connection_t::npos;
        }
        
    private:
        edge_t m_edge;
};

template<size_t NodeCount>
state_t graph_t<NodeCount>::previous(const state_t & state, sigma_t sigma) const {
    edge_t backward_edge = state.e == GRAPH_0 ? switch_setting(sigma, state) : GRAPH_0;
    
    /* Find the vertex connected to the edge backward_edge */
    const typename graph_t<NodeCount>::connection_t *connection = std::find_if(m_graph[state.v], m_graph[state.v] + NodeCount, vertex_contains_edge<NodeCount>(backward_edge));
    
    /*const typename graph_t<NodeCount>::connection_t *connection;
    unsigned int vertex;
    
    for(size_t i = 0; i < NodeCount; ++i) {
        if (m_graph[state.v][i].find(backward_edge) != std::string::npos) {
            connection = &m_graph[state.v][i];
            vertex = i;
            break;
        }
    }*/
    
    unsigned int vertex = connection - m_graph[state.v];
    
    /* Find the edge connecting the newly found vertex to our state vertex */
    return state_t(vertex, m_graph[vertex][state.v][std::find(connection->begin(), connection->end(), backward_edge) - connection->begin()]);
}

template<size_t NodeCount>
state_t graph_t<NodeCount>::next(const state_t & state, sigma_t sigma) const {
    //edge_t backward_edge = state.e == GRAPH_0 ? switch_setting(sigma, state) : GRAPH_0;
    
    /* Find the vertex connected to the edge */
    const typename graph_t<NodeCount>::connection_t *connection = std::find_if(m_graph[state.v], m_graph[state.v] + NodeCount, vertex_contains_edge<NodeCount>(state.e));
    
    unsigned int vertex = connection - m_graph[state.v];
    
    /* Find the edge connecting the newly found vertex to our state vertex */
    edge_t backward_edge = m_graph[vertex][state.v][std::find(connection->begin(), connection->end(), state.e) - connection->begin()];
    
    if (backward_edge == GRAPH_0)
        return state_t(vertex, switch_setting(sigma, state_t(vertex, 0)));
    else
        return state_t(vertex, GRAPH_0);
}

/*
 * probability of a sigma p(s, O|sigma,G)
 * ObservationIteratorT *must* be a reversed iterator
 */

template<size_t NodeCount, typename ObservationIteratorT>
class D : public std::unary_function<sigma_t, probability_t> {
    public:
        inline D(const graph_t<NodeCount> & graph,
                 const state_t & state,
                 const ObservationIteratorT & observation_begin,
                 const ObservationIteratorT & observation_end) : m_graph(graph),
                                                     m_state(state),
                                                     m_observation_begin(observation_begin),
                                                     m_observation_end(observation_end) {
            
        }
        
        probability_t operator() (sigma_t sigma) {
            state_t state = m_state;
            probability_t probability(1);
            
            for(ObservationIteratorT it(m_observation_begin); it != m_observation_end; ++it) {
                edge_t /*backward_edge, edge = switch_setting(sigma, state),*/
                       observation = *it;
                
                //backward_edge = state.e == GRAPH_0 ? edge : GRAPH_0;
                
                probability *= observation == state.e ? (1 - p) : p / 2;
                
                state = m_graph.previous(state, sigma);
            }
            
            return probability;
        }
        
    protected:
        const graph_t<NodeCount> & m_graph;
        state_t m_state;
    private:
        const ObservationIteratorT & m_observation_begin;
        const ObservationIteratorT & m_observation_end;
};

/*
 * p(O | G, sigma) by marginalization over D
 * ObservationIteratorT *must* be a reversed iterator
 */

template<size_t NodeCount, typename ObservationIteratorT>
class observation_probability : public D<NodeCount, ObservationIteratorT> {
    public:
        inline observation_probability(const graph_t<NodeCount> & graph,
                                       const ObservationIteratorT & observation_begin,
                                       const ObservationIteratorT & observation_end) : D<NodeCount, ObservationIteratorT>(graph, state_t(), observation_begin, observation_end) {
            
        }
        
        inline probability_t operator() (sigma_t sigma) {
            probability_t p(0);
            
            for(size_t i = 0; i < NodeCount; ++i) {
                for(size_t j = 0; j < num_edges; ++j) {
                    D<NodeCount, ObservationIteratorT>::m_state.set(i, edges[j]);
                    
                    p += D<NodeCount, ObservationIteratorT>::operator() (sigma);
                }
            }
            
            return p;
        }
};

template<size_t NodeCount>
state_t find_stop_state(const graph_t<NodeCount> & graph, const state_t & start_state, size_t length, sigma_t sigma) {
    state_t state(start_state);
    
    for(size_t i = 0; i < length; ++i)
        state = graph.next(state, sigma);
    
    return state;
}

/*
 * Wrapper for D so that stop states are deduced from start states
 */

template<size_t NodeCount, typename ObservationIteratorT>
class D_over_stop_state : public D<NodeCount, ObservationIteratorT> {
    public:
        inline D_over_stop_state(const graph_t<NodeCount> & graph,
                                 const state_t & start_state,
                                 const ObservationIteratorT & observation_begin,
                                 const ObservationIteratorT & observation_end) : D<NodeCount, ObservationIteratorT>(graph, state_t(), observation_begin, observation_end),
                                                                                 m_graph(graph), m_start_state(start_state), m_observation_begin(observation_begin), m_observation_end(observation_end) {
            
        }
        
        inline probability_t operator() (sigma_t sigma) {
            D<NodeCount, ObservationIteratorT>::m_state = find_stop_state(m_graph, m_start_state, m_observation_end - m_observation_begin, sigma);
            
            return D<NodeCount, ObservationIteratorT>::operator() (sigma);
        }
        
    protected:
        const graph_t<NodeCount> & m_graph;
        state_t m_start_state;
        const ObservationIteratorT & m_observation_begin;
        const ObservationIteratorT & m_observation_end;
};

/*
 * sample over sigmas over all states, pushes samples to sampler functor
 */
 
/*
template<typename VariableT, typename ProbabilityT, typename IteratorT>
class sampler : public std::binary_function<VariableT, ProbabilityT, void> {

template<typename VariableT, typename QFunctorT, typename ProbabilityFunctorT, typename SamplerFunctorT>
void metropolis_hastings(VariableT x, const QFunctorT & Q,
                         const ProbabilityFunctorT & p,
                         const SamplerFunctorT & sampler, size_t num_samples)
*/

template<size_t NodeCount, typename ObservationIteratorT, typename OutputIteratorT>
void sample_sigmas(const graph_t<NodeCount> & graph, const ObservationIteratorT & observation_begin,
                   const ObservationIteratorT & observation_end,
                   OutputIteratorT & output_it, size_t num_samples, size_t burn_in, size_t sampling_interval) {
    sampler<sigma_t, probability_t, OutputIteratorT> s(output_it, burn_in, sampling_interval);
    Q<NodeCount> q;
    
    for(size_t i = 0; i < NodeCount; ++i) {
        for(size_t j = 0; j < num_edges; ++j) {
            sigma_t sigma(random_sigma<NodeCount>());
            D_over_stop_state<NodeCount, ObservationIteratorT> d(graph, state_t(i, edges[j]), observation_begin, observation_end);
            
            // <sigma_t, Q<NodeCount>, D_over_stop_state<NodeCount, ObservationIteratorT>, sampler<sigma_t, probability_t, OutputIteratorT> >
            metropolis_hastings(sigma, q, d, s, num_samples / (NodeCount * num_edges));
        }
    }
}

const size_t smooth_samples = 50;

template<size_t NodeCount, typename ObservationIteratorT, typename OutputIteratorT>
void trace_plot(const graph_t<NodeCount> & graph, const ObservationIteratorT & observation_begin,
                const ObservationIteratorT & observation_end,
                OutputIteratorT & output_it, size_t num_samples) {
    Q<NodeCount> q;
    
    for(size_t i = 0; i < NodeCount; ++i) {
        for(size_t j = 0; j < num_edges; ++j) {
            D_over_stop_state<NodeCount, ObservationIteratorT> d(graph, state_t(i, edges[j]), observation_begin, observation_end);
            std::vector< probability_t > weighted_samples(num_samples / (NodeCount * num_edges), probability_t(0));
            
            for(size_t k = 0; k < smooth_samples; ++k) {
                sigma_t sigma(random_sigma<NodeCount>());
                
                // <sigma_t, Q<NodeCount>, D_over_stop_state<NodeCount, ObservationIteratorT>, sampler<sigma_t, probability_t, OutputIteratorT> >
                
                std::vector< std::pair<sigma_t, probability_t> > samples;
                std::back_insert_iterator< std::vector< std::pair<sigma_t, probability_t> > > inserter(samples);

                sampler<sigma_t, probability_t, std::back_insert_iterator< std::vector< std::pair<sigma_t, probability_t> > > > s(inserter, 0, 1);
                
                metropolis_hastings(sigma, q, d, s, num_samples / (NodeCount * num_edges));
                
                //std::cerr << "asdf4, state: " << i << ", " << edges[j] << ", num_samples: " << samples.size() << " of " << (num_samples / (NodeCount * num_edges)) << std::endl;
                
                for(size_t i = 0; i < samples.size(); ++i)
                    weighted_samples[i] += samples[i].second;
            }
            
            for(size_t i = 0; i < weighted_samples.size(); ++i)
                weighted_samples[i] /= smooth_samples;
            
            output_it = weighted_samples;
                
            ++output_it;
        }
    }
}

/*
 * functor
 */

template<size_t NodeCount, typename ObservationIteratorT>
class state_probability_sigma_functor : public std::binary_function<sigma_t, probability_t, probability_t> {
    public:
        inline state_probability_sigma_functor(const graph_t<NodeCount> & graph,
                                               const state_t & state,
                                               const ObservationIteratorT & observation_begin,
                                               const ObservationIteratorT & observation_end) : m_d(graph, state, observation_begin, observation_end), m_observation_probability(graph, observation_begin, observation_end) {
            
        }
        
        inline probability_t operator() (sigma_t sigma, probability_t p) {
            return m_d(sigma) / m_observation_probability(sigma);
        }
        
    private:
        D<NodeCount, ObservationIteratorT> m_d;
        observation_probability<NodeCount, ObservationIteratorT> m_observation_probability;
};

template<size_t NodeCount, typename ObservationIteratorT, typename SigmaInputIteratorT>
probability_t state_probability(const state_t & state, const graph_t<NodeCount> & graph, const ObservationIteratorT & observation_begin, const ObservationIteratorT & observation_end, const SigmaInputIteratorT & sigmas_begin, const SigmaInputIteratorT & sigmas_end) {
    //return std::accumulate(sigmas_begin, sigmas_end, 0, );
    state_probability_sigma_functor<NodeCount, ObservationIteratorT> f(graph, state, observation_begin, observation_end);
    probability_t p(0);
    
    for(SigmaInputIteratorT it = sigmas_begin; it != sigmas_end; ++it)
        p += f(it->first, it->second);
    
    return p / (sigmas_end - sigmas_begin);
}

/*
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
*/

template<size_t NodeCount, typename ObservationIt>
state_t generate_observation_sequence(const graph_t<NodeCount> & graph, size_t length, ObservationIt it) {
    /* Randomize start position */
    
    sigma_t sigma(random_sigma<NodeCount>());
    
    //std::cerr << "sigma: " << sigma << std::endl;
    
    state_t state(rand() % NodeCount, GRAPH_0);
    
    //std::cerr << "state chosen: " << state.v << std::endl;
    
    for(size_t i = 0; i < length; ++i) {
        state = graph.next(state, sigma);
        
        //std::cerr << "state: " << state << std::endl;
        
        it++ = state.e;
    }
    
    return state;
}

class state_probability_sort_comp : public std::binary_function< std::pair<state_t, probability_t>, std::pair<state_t, probability_t>, bool > {
    public:
        bool operator() (const std::pair<state_t, probability_t> & p1, const std::pair<state_t, probability_t> & p2) const {
            return p1.second < p2.second;
        }
};

int main(int argc, const char **argv) {
    srand(time(NULL));
    const size_t num_samples = 10000, burn_in = 5, sampling_interval = 5;
    
    const size_t NodeCount = 12;
    
    typename graph_t<NodeCount>::connection_t map[][NodeCount] = /*{
        { "", "L", "0", "R" },
        { "0","", "R", "L" },
        { "0", "R","", "L" },
        { "R", "0", "L","" },
    };*/
    {
        {"", "0", "R","","","","","","", "L","","" },
        { "0","", "L", "R","","","","","","","","" },
        { "0", "L","", "R","","","","","","","","" },
        {"", "L", "R","","","","","","","","", "0" },
        {"","","","","", "R","","", "L","", "0","" },
        {"","","","", "R","","", "L", "0","","","" },
        {"","","","","","","", "R","","", "L", "0" },
        {"","","","","", "0", "R","","","", "L","" },
        {"","","","", "R", "L","","","", "0","","" },
        { "0","","","","","","","", "L","","", "R" },
        {"","","","", "R","", "0", "L","","","","" },
        {"","","", "L","","", "R","","", "0","","" },
    };
    
    graph_t<NodeCount> graph;
    graph.parse(map);
    
    const size_t observation_length = 100;
    
    typedef std::vector<edge_t> observation_sequence_t;

    observation_sequence_t observation;
    const state_t state(generate_observation_sequence<NodeCount>(graph, observation_length, std::back_insert_iterator<observation_sequence_t>(observation)));
    
    std::cerr << "observation sequence:";
    std::copy(observation.begin(), observation.end(), std::ostream_iterator<edge_t>(std::cerr, ", "));
    std::cerr << "stop state is: " << state << std::endl;

    //std::cerr << "previous:" << graph.previous(state_t(0, '0'), 0) << std::endl;
    
    typedef std::vector< std::pair<sigma_t, probability_t> > sigma_sequence_t;
    
    sigma_sequence_t sigmas;
    
    std::back_insert_iterator<sigma_sequence_t> inserter(sigmas);
    sample_sigmas<NodeCount>(graph, observation.begin(), observation.end(), inserter, num_samples, burn_in, sampling_interval);
    
    /*std::cerr << "sigmas:" << std::endl;
    probability_t p(0);
    for(sigma_sequence_t::iterator it = sigmas.begin(); it != sigmas.end(); ++it) {
        std::cerr << "(" << it->first << ", " << it->second << "), ";
        p += it->second;
    }*/
    
    //std::cerr << "state_probability: " << state_probability<NodeCount>(state, graph, observation.begin(), observation.end(), sigmas.begin(), sigmas.end()) << std::endl;
    
    std::vector< std::pair<state_t, probability_t> > sorted_states;
    
    probability_t total_prob(0);
    for(size_t i = 0; i < NodeCount; ++i) {
        for(size_t j = 0; j < num_edges; ++j) {
            probability_t prob = state_probability<NodeCount>(state_t(i, edges[j]), graph, observation.begin(), observation.end(), sigmas.begin(), sigmas.end());
            //std::cerr << "state_probability: " << prob << " for stop_state: " << state_t(i, edges[j]) << std::endl;
            total_prob += prob;
            
            sorted_states.push_back(std::make_pair(state_t(i, edges[j]), prob));
        }
    }
    
    std::cerr << "total_prob: " << total_prob << std::endl;
    
    std::sort(sorted_states.begin(), sorted_states.end(), state_probability_sort_comp());
    
    for(std::vector< std::pair<state_t, probability_t> >::const_iterator it = sorted_states.begin(); it != sorted_states.end(); ++it) {
        std::cerr << "probability: " << it->second << " for state: " << it->first;
        
        if (it->first.v == state.v && it->first.e == state.e)
            std::cerr << " [ answer ]";
        std::cerr << std::endl;
    }
    
    /*
     * trace plot
     */
                
    std::vector<std::vector< probability_t > > traces;
    std::back_insert_iterator< std::vector<std::vector< probability_t > > > traces_inserter(traces);
    trace_plot(graph, observation.begin(), observation.end(), traces_inserter, num_samples);
    
    //std::cerr << "plot done" << std::endl;
    
    std::fstream file("plot.py", std::ios_base::out);
    
    /*
    pl.semilogy([ prob / length for prob in p ])
        #break
        
    pl.xlabel('interval')
    pl.ylabel('log(probability) + k')
    pl.title('Convergence plot')
    pl.grid(True)
    pl.savefig(filename, bbox_inches = 0)
    
    pl.show()
    */
    
    file << "import pylab as pl" << std::endl << "pl.xlabel('interval')\npl.ylabel('log(probability) + k')\npl.title('Convergence plot')\npl.grid(True)" << std::endl;
    
    for(std::vector<std::vector< probability_t > >::const_iterator it = traces.begin(); it != traces.end(); ++it) {
        file << "pl.semilogy([ ";
        
        for(std::vector< probability_t >::const_iterator jt = it->begin(); jt != it->end() -1; ++jt)
            file << *jt << ", ";
        
        file << *(it->end() - 1) << " ])" << std::endl;
    }
    
    file << "pl.show()" << std::endl;
    
    file.close();
}


