#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "alglib/dataanalysis.h"
#include <armadillo>

using namespace std;
using namespace alglib;
using namespace arma;

typedef struct cluster_report {
/*************************************************************************
This structure saves the information about a K-Means instance.

Variables:
    centroids   -   mat (structure from Armadillo library).
    k           -   the number of clusters.
    BetaCV      -   a metric (double value) used to choose the optimal
                    number of clusters through the Elbow criterion.
    cidx        -   matrix [1, n_data] indicating the membership of data
                    to a cluster through an index (this information
                    is generated with create_cidx_matrix function).
*************************************************************************/
    mat centroids;
    int k;
    double BetaCV;
    int *cidx;
} cluster_report;

extern void getDatasetDims(string fname, int *dim, int *data);
extern void loadData(string fname, double **array, int n_dims);
extern int cluster_size(cluster_report rep, int cluster_id, int n_data);
extern void create_cidx_matrix(double **data, int n_data, cluster_report instance);
extern double L2distance(double xc, double yc, double x1, double y1);
extern void data_out(double ****data, long *lastitem, string name, bool ***incircle, int peers, int cs, cluster_report *report);
