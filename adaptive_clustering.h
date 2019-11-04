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
