#ifndef DISTRIBUTEDADAPTIVECLUSTERING_GRAPH_H
#define DISTRIBUTEDADAPTIVECLUSTERING_GRAPH_H
#include <igraph/igraph.h>
#include <iostream>

/**
 * @file graph.h
 */

using namespace std;

/**
 * This function generates a geometric random graph by dropping points (vertices)
 * randomly to the unit square and then connecting all those pairs
 * which are less than radius apart in Euclidean norm.
 *
 * @param [in] n - The number of vertices in the graph
 * @param [in] radius - The radius within which the vertices will be connected
 * @return an igraph_t structure (from igraph) indicating the graph.
 */
igraph_t generateGeometricGraph(igraph_integer_t n, igraph_real_t radius);
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
igraph_t generateBarabasiAlbertGraph(igraph_integer_t n, igraph_real_t power, igraph_integer_t m, igraph_real_t A);
/**
 * This function generates a random (Erdos-Renyi) graph
 *
 * @param [in] n - The number of nodes in the generated graph
 * @param [in] type - The type of the random graph
 * @param [in] param - This is the p parameter for G(n,p) graphs and the m parameter for G(n,m) graphs
 * @return an igraph_t structure (from igraph) indicating the graph.
 */

igraph_t generateErdosRenyiGraph(igraph_integer_t n, igraph_erdos_renyi_t type, igraph_real_t param);
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
igraph_t generateRegularGraph(igraph_integer_t n, igraph_integer_t k);
/**
 * This function generates the selected random graph
 *
 * @param [in] type - The type of the random graph: 1 geometric, 2 Barabasi-Albert, 3 Erdos-Renyi, 4 regular (clique)
 * @param [in] n - The number of nodes in the generated graph
 * @return an igraph_t structure (from igraph) indicating the graph.
 */
igraph_t generateRandomGraph(int type, int n);
/**
 * Print the graph type
 *
 * @param [in] type - The type of the random graph
 */
void printGraphType(int type);
/**
 * This function prints on terminal tha minimum and maximum vertex degree
 * @param graph the graph structure (from igraph).
 * @param outputOnFile a boolean value for print purpose.
 * @return an igraph_vector_t structure (from igraph) containing the requested informations.
 */
igraph_vector_t getMinMaxVertexDeg(igraph_t graph, bool outputOnFile);

#endif //DISTRIBUTEDADAPTIVECLUSTERING_GRAPH_H
