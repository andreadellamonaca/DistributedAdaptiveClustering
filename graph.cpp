#include "graph.h"

/**
 * This function generates a geometric random graph by dropping points (vertices)
 * randomly to the unit square and then connecting all those pairs
 * which are less than radius apart in Euclidean norm.
 *
 * @param [in] n - The number of vertices in the graph
 * @param [in] radius - The radius within which the vertices will be connected
 * @return an igraph_t structure (from igraph) indicating the graph.
 */
igraph_t generateGeometricGraph(igraph_integer_t n, igraph_real_t radius)
{
    igraph_t G_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the geometric model
    igraph_grg_game(&G_graph, n, radius, 0, 0, 0);

    igraph_is_connected(&G_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&G_graph);
        igraph_grg_game(&G_graph, n, radius, 0, 0, 0);

        igraph_is_connected(&G_graph, &connected, IGRAPH_WEAK);
    }
    return G_graph;
}

/**
 * This function generates a graph based on the Barabási-Albert model.
 *
 * @param [in] n - The number of vertices in the graph
 * @param [in] power - Power of the preferential attachment. The probability that a cited vertex
 *                      is proportional to d^power+A, where d is its degree (see also the outpref argument),
 *                      power and A are given by arguments. In the classic preferential attachment model power=1.
 * @param [in] m - The number of outgoing edges generated for each vertex. (Only if outseq is NULL.)
 * @param [in] A - The probability that a cited vertex is proportional to d^power+A, where d is its degree
 *                  (see also the outpref argument), power and A are given by arguments.
 *                  In the previous versions of the function this parameter was implicitly set to one.
 * @return an igraph_t structure (from igraph) indicating the graph.
 */
igraph_t generateBarabasiAlbertGraph(igraph_integer_t n, igraph_real_t power, igraph_integer_t m, igraph_real_t A)
{
    // n = The number of vertices in the graph
    // power = Power of the preferential attachment. The probability that a vertex is cited is proportional to d^power+A, where d is its degree, power and A are given by arguments. In the classic preferential attachment model power=1
    // m = number of outgoing edges generated for each vertex
    // A = The probability that a vertex is cited is proportional to d^power+A, where d is its degree, power and A are given by arguments

    igraph_t BA_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the Barabasi-Albert model
    igraph_barabasi_game(/* graph=    */ &BA_graph,
            /* n=        */ n,
            /* power=    */ power,
            /* m=        */ m,
            /* outseq=   */ 0,
            /* outpref=  */ 0,
            /* A=        */ A,
            /* directed= */ IGRAPH_UNDIRECTED,
            /* algo=     */ IGRAPH_BARABASI_PSUMTREE,
            /* start_from= */ 0);

    igraph_is_connected(&BA_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&BA_graph);
        igraph_barabasi_game(/* graph=    */ &BA_graph,
                /* n=        */ n,
                /* power=    */ power,
                /* m=        */ m,
                /* outseq=   */ 0,
                /* outpref=  */ 0,
                /* A=        */ A,
                /* directed= */ IGRAPH_UNDIRECTED,
                /* algo=     */ IGRAPH_BARABASI_PSUMTREE,
                /* start_from= */ 0);

        igraph_is_connected(&BA_graph, &connected, IGRAPH_WEAK);
    }
    return BA_graph;
}

/**
 * This function generates a random (Erdos-Renyi) graph
 *
 * @param [in] n - The number of nodes in the generated graph
 * @param [in] type - The type of the random graph
 * @param [in] param - This is the p parameter for G(n,p) graphs and the m parameter for G(n,m) graphs
 * @return an igraph_t structure (from igraph) indicating the graph.
 */
igraph_t generateErdosRenyiGraph(igraph_integer_t n, igraph_erdos_renyi_t type, igraph_real_t param)
{
    // n = The number of vertices in the graph
    // type = IGRAPH_ERDOS_RENYI_GNM G(n,m) graph, m edges are selected uniformly randomly in a graph with n vertices.
    //      = IGRAPH_ERDOS_RENYI_GNP G(n,p) graph, every possible edge is included in the graph with probability p

    igraph_t ER_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the Erdos-Renyi model
    igraph_erdos_renyi_game(&ER_graph, type, n, param, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);

    igraph_is_connected(&ER_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&ER_graph);
        igraph_erdos_renyi_game(&ER_graph, type, n, param, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);

        igraph_is_connected(&ER_graph, &connected, IGRAPH_WEAK);
    }
    return ER_graph;
}

/**
 * This function generates a directed or undirected random graph where
 * the degrees of vertices are equal to a predefined constant k.
 * For undirected graphs, at least one of k and the number of vertices
 * must be even.
 *
 * @param [in] n - The number of nodes in the generated graph
 * @param [in] k - The degree of each vertex in an undirected graph, or the out-degree and in-degree of each
 *                  vertex in a directed graph
 * @return an igraph_t structure (from igraph) indicating the graph.
 */
igraph_t generateRegularGraph(igraph_integer_t n, igraph_integer_t k)
{
    // n = The number of vertices in the graph
    // k = The degree of each vertex in an undirected graph. For undirected graphs, at least one of k and the number of vertices must be even.

    igraph_t R_graph;
    igraph_bool_t connected;

    // generate a connected regular random graph
    igraph_k_regular_game(&R_graph, n, k, IGRAPH_UNDIRECTED, 0);

    igraph_is_connected(&R_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&R_graph);
        igraph_k_regular_game(&R_graph, n, k, IGRAPH_UNDIRECTED, 0);

        igraph_is_connected(&R_graph, &connected, IGRAPH_WEAK);
    }
    return R_graph;
}

/**
 * This function generate the selected random graph
 *
 * @param [in] type - The type of the random graph: 1 geometric, 2 Barabasi-Albert, 3 Erdos-Renyi, 4 regular (clique)
 * @param [in] n - The number of nodes in the generated graph
 * @return an igraph_t structure (from igraph) indicating the graph.
 */
igraph_t generateRandomGraph(int type, int n)
{
    // turn on attribute handling in igraph
    igraph_i_set_attribute_table(&igraph_cattribute_table);
    // seed igraph PRNG
    igraph_rng_seed(igraph_rng_default(), 42);
    igraph_t random_graph;

    switch (type) {
        case 1:
            random_graph = generateGeometricGraph(n, sqrt(100.0/(float)n));
            break;
        case 2:
            random_graph = generateBarabasiAlbertGraph(n, 1.0, 5, 1.0);
            break;
        case 3:
            random_graph = generateErdosRenyiGraph(n, IGRAPH_ERDOS_RENYI_GNP, 10.0/(float)n);
            // random_graph = generateErdosRenyiGraph(n, IGRAPH_ERDOS_RENYI_GNM, ceil(n^2/3));
            break;
        case 4:
            random_graph = generateRegularGraph(n, n-1);
            break;
        default:
            break;
    }
    return random_graph;
}

/**
 * Print the graph type
 *
 * @param [in] type - The type of the random graph
 */
void printGraphType(int type)
{
    switch (type) {
        case 1:
            cout << "Geometric random graph" << endl;
            break;
        case 2:
            cout << "Barabasi-Albert random graph" << endl;
            break;
        case 3:
            cout << "Erdos-Renyi random graph" << endl;
            break;
        case 4:
            cout << "Regular random graph" << endl;
            break;
        default:
            break;
    }
}

/**
 * This function print on terminal tha minimum and maximum vertex degree
 * @param graph the graph structure (from igraph).
 * @param outputOnFile a boolean value for print purpose.
 * @return an igraph_vector_t structure (from igraph) containing the requested informations.
 */
igraph_vector_t getMinMaxVertexDeg(igraph_t graph, bool outputOnFile) {
    igraph_vector_t result;
    igraph_real_t mindeg;
    igraph_real_t maxdeg;

    igraph_vector_init(&result, 0);
    igraph_degree(&graph, &result, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
    igraph_vector_minmax(&result, &mindeg, &maxdeg);
    if (!outputOnFile) {
        cout << "Minimum degree is " << (int) mindeg << ", Maximum degree is " << (int) maxdeg << endl;
    }
    return result;
}