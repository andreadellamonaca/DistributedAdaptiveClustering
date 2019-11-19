#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <armadillo>

/**
 * @file adaptive_clustering.h
 */

using namespace std;
using namespace arma;

/**
 * @struct cluster_report
 */
typedef struct cluster_report {
    mat centroids;      /**< Mat (structure from Armadillo library). */
    int k;              /**< The number of clusters. */
    double BetaCV;      /**< A metric (double value) used to choose the optimal number of clusters through the Elbow criterion. */
    int *cidx;          /**< A matrix [1, n_data] indicating the membership of data to a cluster through an index. @see create_cidx_matrix for matrix generation */
} cluster_report;

extern int getDatasetDims(string fname, int &dim, int &data);
extern int loadData(string fname, double **array, int n_dims);
extern int cluster_size(cluster_report rep, int cluster_id, int n_data);
extern int mindistCluster(mat centroids, double first_coordinate, double second_coordinate);
extern int create_cidx_matrix(double **data, int partitionSize, cluster_report &instance);
extern double L2distance(double xc, double yc, double x1, double y1);
extern void data_out(double ****data, long *lastitem, string name, bool ***incircle, int peers, int cs, cluster_report *report);
