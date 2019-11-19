#ifndef DISTRIBUTEDADAPTIVECLUSTERING_GRAPH_H
#define DISTRIBUTEDADAPTIVECLUSTERING_GRAPH_H
#include <igraph/igraph.h>
#include <iostream>

/**
 * @file graph.h
 */

using namespace std;

igraph_t generateGeometricGraph(igraph_integer_t n, igraph_real_t radius);
igraph_t generateBarabasiAlbertGraph(igraph_integer_t n, igraph_real_t power, igraph_integer_t m, igraph_real_t A);
igraph_t generateErdosRenyiGraph(igraph_integer_t n, igraph_erdos_renyi_t type, igraph_real_t param);
igraph_t generateRegularGraph(igraph_integer_t n, igraph_integer_t k);
igraph_t generateRandomGraph(int type, int n);
void printGraphType(int type);
igraph_vector_t getMinMaxVertexDeg(igraph_t graph, bool outputOnFile);


#endif //DISTRIBUTEDADAPTIVECLUSTERING_GRAPH_H