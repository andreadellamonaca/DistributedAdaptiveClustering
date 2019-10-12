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

#define K_MAX 10 //Max number of clusters for Elbow criterion
#define ELBOW_THRES 0.25 //BetaCV Threshold for Elbow criterion
#define PERCENTAGE_INCIRCLE 0.90 //Percentage of points within the circle
#define PERCENTAGE_SUBSPACES 0.80 //Percentage of subspaces for outliers occurrences evaluation

//string filename = "../Iris.csv";
//string filename = "../HTRU_2.csv";
//string filename = "../dataset_benchmark/dim032.csv";
//string filename = "../Absenteeism_at_work.csv";

//bool save_output = false; //Flag for csv output generation
//string outdir = "../plot/dim032/";

typedef struct AVG_SUMMARY {
    double *local;
    int pts_count;
} AVG_SUMMARY;

typedef struct cluster_report {
    mat centroids;
    int k;
    double BetaCV;
    int *cidx;
} cluster_report;

extern void getDatasetDims(string fname, int *dim, int *data);
extern void loadData(string fname, double **array, int n_dims);
extern double getMean(double *arr, int n_data);
extern void Standardize_dataset(double **data, int n_dims, int n_data);
extern double PearsonCoefficient(double *X, double *Y, int n_data);
extern void PCA_transform(double **data_to_transform, int data_dim, int n_data, double **new_space);
extern int cluster_size(cluster_report rep, int cluster_id, int n_data);
extern cluster_report run_K_means(double **data_to_transform, int n_data);
extern void create_cidx_matrix(double **data, int n_data, cluster_report instance);
extern double WithinClusterSS(double **data, cluster_report instance, int n_data);
extern double BetweenClusterSS(double **data, cluster_report instance, int n_data);
extern int WithinClusterPairs(cluster_report instance, int n_data);
extern int BetweenClusterPairs(cluster_report instance, int n_data);
extern double BetaCV(double **data, cluster_report instance, int n_data);
extern double L2distance(double xc, double yc, double x1, double y1);
extern void csv_out_info(double **data, int n_data, string outdir, string name, bool *incircle, cluster_report report);
