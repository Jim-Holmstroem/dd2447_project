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

/*
 * The SGI std::identity newer made it to the std :(
 * However it's kinda easy to implement :)
 */

template<typename T>
struct identity : public std::unary_function<T, T> {
    public:
    
    inline const T & operator() (const T & t) const {
        return t;
    }
};

/*
 * Generic templated fast randomize element from range functor.
 * Used to randomize from available vertices but also to find labels
 * on switches
 * operator is a functor following std::unary_operator<ElementT, bool>
 * whose operator() return true if the element is available for usage
 * Note that it requires the tracking of how many items in the range
 * are already occupied
 */

template<typename IteratorT, typename OperatorT, typename SizeT>
IteratorT random_index(IteratorT it, const IteratorT & end, SizeT available_indices_count, const OperatorT & op) {
    SizeT index = rand() % available_indices_count;
    
    for(; it != end; ++it) {
        if (op(*it)) {
            if (index-- <= 0)
                return it;
        }
    }
    
    return end;
}

/* For conversion between int to labels */
static const char connection_types[] = "LR0";

template<size_t VertexCount, bool AllowLoops = true>
class random_graph_t {
    public:
        typedef char connection_type_t;
        typedef std::vector<connection_type_t> connection_t;
        
        /*
         * Quite the ugly wrapper for a vector but is required as the templating
         * requires a unique type
         */
        
        /*struct connection_t {
            public:
                typedef std::vector<connection_type_t>::const_iterator const_iterator;
                typedef std::vector<connection_type_t>::iterator iterator;
                
                inline std::vector<connection_type_t>::const_iterator begin() const {
                    return m_connection.begin();
                }
                
                inline std::vector<connection_type_t>::const_iterator end() const {
                    return m_connection.end();
                }
                
                inline std::vector<connection_type_t>::iterator begin() {
                    return m_connection.begin();
                }
                
                inline std::vector<connection_type_t>::iterator end() {
                    return m_connection.end();
                }
                
                inline bool empty() const {
                    return m_connection.empty();
                }
                
                inline std::vector<connection_type_t>::size_type size() const {
                    return m_connection.size();
                }
                
                inline void push_back(connection_type_t conn) {
                    m_connection.push_back(conn);
                }
                
            private:
            std::vector<connection_type_t> m_connection;
        };*/
        
        typedef std::pair<unsigned int, unsigned int> edge_t;
        
        static const size_t size = VertexCount;
        static const size_t connections_per_vertex;
        
        enum {
            None = '\0',
            L = 'L',
            R = 'R',
            O = '0'
        };
        
        const connection_t *operator[] (size_t index) const {
            return m_graph[index];
        }
        
        /*
         * randomize a valid graph
         */
        
        random_graph_t() : m_graph(), m_fully_occupied_vertices_count(0) {
            memset(m_connections_per_vertex, 0, sizeof(m_connections_per_vertex));
            
            while (m_fully_occupied_vertices_count < size)
                assign_edge(random_possible_edge());
            
            /*
             * Now label edges
             */
            
            for(size_t i = 0; i < size; ++i) {
                bool label_available[] = { true, true, true };
                unsigned int labels_available_count = sizeof(label_available) / sizeof(bool);
                
                for(size_t j = 0; j < size; ++j) {
                    for(typename connection_t::iterator it = m_graph[i][j].begin(); it != m_graph[i][j].end(); ++it) {
                        bool *rit = random_index(label_available,
                                            label_available + sizeof(label_available) / sizeof(bool),
                                            labels_available_count,
                                            identity<bool>());

                        *rit = false;
                        *it = connection_types[rit - label_available];
                        --labels_available_count;
                    }
                }
            }
        }
        
        /*
         * Debugging
         */
        
        
        /*
         * Really simple graph plot, just randomizes coordinates over a canvas then draws lines between them in a random color
         * and render the L, R or 0 type of the switch
         */
        
        std::ostream & svg(std::ostream & stream) const {
            return stream;
        }
        
        bool validate() const {
            /*
             * Make sure the graph is symmetric in the way connections are defined
             * in both directions, note that the labels do *NOT* have to match!
             */
            
            for(size_t i = 0; i < size; ++i) {
                for(size_t j = 0; j < size; ++j) {
                    if (m_graph[i][j].size() != m_graph[j][i].size()) {
                        std::cerr << "graph[" << i << "][" << j << "] != graph[" << j << "][" << i << "]" << std::endl;
                        return false;
                    }
                }
            }
            
            /*
             * Make sure connection count is the same
             */
            
            for(size_t i = 0; i < size; ++i) {
                if (m_connections_per_vertex[i] != connections_per_vertex) {
                    std::cerr << "connections[" << i << "]: " << m_connections_per_vertex[i] << " != " << connections_per_vertex << std::endl;
                    return false;
                }
            }
            
            if (m_fully_occupied_vertices_count != size) {
                std::cerr << "fully_occupied_vertices_count != size" << std::endl;
                return false;
            }
            
            return true;
        }
        
    private:
        connection_t m_graph [VertexCount][VertexCount];
        unsigned int m_connections_per_vertex[VertexCount], m_fully_occupied_vertices_count;
        
        /*
         * Find an available vertex
         * Old version does a plain randomize and iterates until successful,
         * this can miss a lot, especially in the end.
         * The new instead randomizes only of the set of available
         * vertices, at the cost of a linear seek
         */
        
        unsigned int random_available_vertex() const {
            const unsigned int *it = 
                random_index(m_connections_per_vertex,
                             m_connections_per_vertex + size,
                             size - m_fully_occupied_vertices_count,
                             std::bind2nd(std::not_equal_to<unsigned int>(),
                             connections_per_vertex));
            
            assert(it != m_connections_per_vertex + size);
            
            return (unsigned int) (it - m_connections_per_vertex);
        }
        
        /*
         * The number of connections from a vertex
         */
        
        inline size_t vertex_connection_count(unsigned int index) const {
            /* Debugging consistency */
            #if 1
            size_t count = 0;
            
            for(size_t i = 0; i < size; ++i)
                count += m_graph[index][i].size();
            
            assert(count == m_connections_per_vertex[index]);
            
            #endif
            
            return m_connections_per_vertex[index];
        }
        
        /* Find a possible edge */
        
        edge_t random_possible_edge() const {
            /* Template argument, will be eliminated on compile */
            
            if (!AllowLoops) {
                unsigned int v1 = random_available_vertex(), v2;
                
                do {
                    v2 = random_available_vertex();
                }
                while (v2 == v1);
                
                return std::make_pair(v1, v2);
            }
            else {
                /*
                 * There is a special case, if the two vertices chosen
                 * are the same, it is allowed, however, if these already have
                 * two edges connected, it will not be possible to make a
                 * connection to itself, because that will occupy two "slots"
                 */
                unsigned int v1 = random_available_vertex(),
                             v2 = random_available_vertex();
                             
                if (v1 == v2) {
                    if (vertex_connection_count(v1) > connections_per_vertex - 2) {
                        /*
                         * Even though the vertex can take another connection,
                         * there's not space for two!
                         */
                        
                        /* Special case: If there are other loops,
                         * and even if the number of vertices is even
                         * we can have a case where only a single vertex is left
                         * and there is no solution,
                         * in this case we abort, but we could do a repair
                         * easier is to disable loops
                         */
                        
                        if (m_fully_occupied_vertices_count == size - 1) {
                            std::cerr << "There's only one node remaining and "
                                         "there are probably loops occuring "
                                         "but this node can't handle another loop, "
                                         "it has only space for one connection. "
                                         "This is a very rare occasion and can be remedied "
                                         "but i'm too lazy to do anything about it :)" << std::endl;

                            assert(m_fully_occupied_vertices_count < size - 1);
                        }
                        
                        do {
                            v2 = random_available_vertex();
                        }
                        while (v2 == v1);
                    }
                }
                
                return std::make_pair(v1, v2);
            }
        }
        
        /* Create the edge */
        
        void assign_edge(const edge_t & edge) {
            if (edge.first == edge.second) {
                assert(vertex_connection_count(edge.first) <= connections_per_vertex - 2);
            }
            else {
                assert(vertex_connection_count(edge.first) < connections_per_vertex);
                assert(vertex_connection_count(edge.second) < connections_per_vertex);
            }
            
            /* Push back blank connection */
            
            m_graph[edge.first][edge.second].push_back(None);
            m_graph[edge.second][edge.first].push_back(None);
            
            /* 
             * Track connections per vertex and fully occupied vertices, 
             * makes for faster randomizing of vertices/edges
             */
            
            ++m_connections_per_vertex[edge.first];
            ++m_connections_per_vertex[edge.second];
            
            if (m_connections_per_vertex[edge.first] >= connections_per_vertex)
                ++m_fully_occupied_vertices_count;
            
            if (m_connections_per_vertex[edge.second] >= connections_per_vertex) {
                if (edge.first != edge.second)
                    ++m_fully_occupied_vertices_count;
            }
        }
};

/*
 * Prints a Python-style tuple representation of the graph
 */

/*template<size_t VertexCount, bool AllowLoops>
std::ostream & operator<<(std::ostream & stream, const typename random_graph_t<VertexCount, AllowLoops>::connection_t & connections) {
    if (connections.empty())
        stream << '0';
    else {
        stream << "( ";
        
        std::copy(connections.begin(),
                  connections.end(),
                  std::ostream_iterator< typename random_graph_t<VertexCount, AllowLoops>::connection_type_t > (stream, ", "));
        
        stream << " )";
    }
}*/

template<size_t VertexCount, bool AllowLoops>
std::ostream & operator<<(std::ostream & stream, const random_graph_t<VertexCount, AllowLoops> & graph) {
    stream << '{' << std::endl;
    
    for(size_t row = 0; row < random_graph_t<VertexCount, AllowLoops>::size; ++row) {
        stream << '{';

        for(size_t col = 0; col < random_graph_t<VertexCount, AllowLoops>::size; ++col) {
            if (graph[row][col].empty())
                stream << "\"\"";
            else {
                stream << " \"";
                
                typename random_graph_t<VertexCount, AllowLoops>::connection_t::const_iterator it;
                for(it = graph[row][col].begin(); it != graph[row][col].end() - 1; ++it) {
                    //stream << " '" << *it << "',";
                    stream << *it;
                }
                
                stream << *it << "\"";
            }
            
            stream << (col != random_graph_t<VertexCount, AllowLoops>::size - 1 ? ',' : ' ');
        }

        stream << '}' << (row != random_graph_t<VertexCount, AllowLoops>::size ? ',' : ' ') << std::endl;
    }
    
    stream << '}';
    
    return stream;
}

template<size_t VertexCount, bool AllowLoops>
const size_t random_graph_t<VertexCount, AllowLoops>::connections_per_vertex(3);

int main(int argc, const char **argv) {
    srand(time(NULL));
    
    static const size_t vertex_count = 12;
    static const bool allow_loops = false;
        
    /* Too large graphs won't fit nice on the stack, thus use the heap */
    random_graph_t<vertex_count, allow_loops> *graph = new random_graph_t<vertex_count, allow_loops>();
    
    std::cerr << "Validate: " << std::boolalpha << graph->validate() << std::endl;
    
    std::cerr << "Array dump: " << std::endl << *graph << std::endl;
}

