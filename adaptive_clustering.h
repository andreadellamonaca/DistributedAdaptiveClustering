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

/**
 * The algorithm calculates the number of rows and columns in a CSV file and stores the information into dim and data.
 * @param [in] fname the path referred to a CSV file.
 * @param [in,out] dim the number of columns in the CSV file.
 * @param [in,out] data the number of rows in the CSV file.
 * @return 0 if file is read correctly, otherwise -9;
 */
extern int getDatasetDims(string fname, int &dim, int &data);
/**
 * The algorithm loads a CSV file into a matrix of double values.
 * @param [in] fname the number of columns in the CSV file.
 * @param [in,out] array a matrix [n_data, n_dims], in which the CSV file is loaded.
 * @param [in] n_dims the number of columns in the CSV file.
 * @return 0 if the dataset is loaded correctly, -2 if array is NULL,
 *          -8 if there is a conversion error on a read value, -9 if
 *          there is an error with inputFile.
 */
extern int loadData(string fname, double **array, int n_dims);
/**
 * The algorithm calculates the number of elements of a specified cluster in the given cluster report.
 * @param [in] rep a cluster_report structure describing the K-Means instance carried out.
 * @param [in] cluster_id a number (between 0 and rep.k) indicating the cluster whereby we want to know the number of data in it.
 * @param [in] n_data number of rows in the dataset matrix.
 * @return an integer indicating the number of data in the cluster with the given id.
 *          The program exit with -2 if rep.cidx is NULL.
 */
extern int cluster_size(cluster_report rep, int cluster_id, int n_data);
/**
 * This function finds the cluster with the minimum distance to a given point.
 * @param [in] centroids mat structure [2, nClusters] (from Armadillo).
 * @param [in] first_coordinate the first component of the point.
 * @param [in] second_coordinate the second component of the point.
 * @return an index between 0 and (centroids.n_cols-1). The program exits with -2 if centroids is empty.
 */
extern int mindistCluster(mat centroids, double first_coordinate, double second_coordinate);
/**
 * The algorithm creates the array [1, N_DATA] which indicates the membership of each data to a cluster through an integer.
 * @param [in] data a matrix [n_data, n_dims] on which the K-Means run was made.
 * @param [in] partitionSize number of rows in the dataset matrix.
 * @param [in,out] instance a cluster_report structure describing the K-Means instance carried out.
 * @return 0 if success, -2 if data or cidx are NULL or centroids is empty.
 */
extern int create_cidx_matrix(double **data, int partitionSize, cluster_report &instance);
/**
 * The algorithm calculates the Euclidean distance between two points p_c and p_1.
 * @param [in] xc the first component of p_c.
 * @param [in] yc the second component of p_c.
 * @param [in] x1 the first component of p_1.
 * @param [in] y1 the second component of p_1.
 * @return a double value indicating the Euclidean distance between the two given points.
 */
extern double L2distance(double xc, double yc, double x1, double y1);
/**
 * This function partitions the data between the peers and it generates 2 vectors:
 * peerLastItem has the index of last item for each peer, while partitionSize has
 * the count of elements that each peer manages.
 * @param [in] n_data the number of data to partition.
 * @param [in] peers the number of peers.
 * @param [in,out] peerLastItem a long array [1, peers]
 * @param [in,out] partitionSize a long array [1, peers]
 * @return 0 if it is correct, -1 for memory allocation error on peerLastItem or
 *          partitionSize, otherwise -7.
 */
int partitionData(int n_data, int peers, long **peerLastItem, long **partitionSize);
/**
 * This function computes the average on each dimension for a single peer.
 * @param [in] data a matrix [n_data, n_dims] containing the dataset.
 * @param [in] ndims the number of dimensions.
 * @param [in] start the first element of the peer.
 * @param [in] end the last element of the peer.
 * @param [in,out] summaries an array [1, n_dims] in which the average of each dimension is stored.
 * @return 0 if it is correct, -2 if data or summaries are NULL.
 */
int computeLocalAverage(double **data, int ndims, long start, long end, double *summaries);
/**
 * This function centers the dataset on each dimension for a single peer.
 * @param [in] summaries an array [1, n_dims] containing the average of each dimension.
 * @param [in] ndims the number of dimensions.
 * @param [in] start the first element of the peer.
 * @param [in] end the last element of the peer.
 * @param [in,out] data a matrix [n_data, n_dims] containing the dataset to be centered.
 * @return 0 if it is correct, -2 if data or summaries are NULL.
 */
int CenterData(double *summaries, int ndims, long start, long end, double **data);
/**
 * This function computes the Pearson Matrix on the local dataset of each peer.
 * The covariance is stored on pcc, while the standard deviation of each dimension
 * is stored in squaresum_dims.
 * @param [in,out] pcc a matrix [ndims, ndims] to store the covariance between each pair
 *                      of dimensions.
 * @param [in,out] squaresum_dims an array [1, ndims] to store the standard deviation of
 *                                  each dimension.
 * @param [in] ndims the number of dimensions.
 * @param [in] start the first element of the peer.
 * @param [in] end the last element of the peer.
 * @param [in] data a matrix [n_data, n_dims] containing the dataset to be used.
 * @return 0 if it is correct, -2 if pcc, squaresum_dims or data are NULL.
 */
int computeLocalPCC(double **pcc, double *squaresum_dims, int ndims, long start, long end, double **data);
/**
 * This function computes the Pearson matrix for each peer dividing the covariance
 * stored in pcc by squaresum_dims.
 * @param [in,out] pcc a matrix [ndims, ndims] containing the covariance between
 *                      each pair of dimensions.
 * @param [in] squaresum_dims an array [1, ndims] containing the standard deviation of
 *                            each dimension.
 * @param [in] ndims the number of dimensions.
 * @return 0 if it is correct, -2 if pcc or squaresum_dims are NULL.
 */
int computePearsonMatrix(double **pcc, double *squaresum_dims, int ndims);
/**
 * This function says if a dimension must belong to the CORR set computing
 * the overall Pearson coefficient (row-wise sum of the Pearson coefficients)
 * for a given dimension. The function exits with code -2 if pcc is NULL.
 * @param [in] ndims the number of dimensions.
 * @param [in] dimensionID an index between 0 and ndims indicating the given dimension.
 * @param [in] pcc the matrix [ndims, ndims] containing the Pearson coefficients.
 * @return true if the dimension must belong to CORR, false otherwise.
 */
bool isCorrDimension(int ndims, int dimensionID, double **pcc);
/**
 * This function computes the number of dimensions that belong to the CORR
 * and UNCORR sets.
 * @param [in] pcc the matrix [ndims, ndims] containing the Pearson coefficients.
 * @param [in] ndims the number of dimensions.
 * @param [in,out] corr_vars an integer value which will contain the cardinality
 *                           of CORR.
 * @param [in,out] uncorr_vars an integer value which will contain the
 *                              cardinality of UNCORR.
 * @return 0 if it is correct, -2 if pcc is NULL.
 */
int computeCorrUncorrCardinality(double **pcc, int ndims, int &corr_vars, int &uncorr_vars);
/**
 * This function copies a single peer partition of the matrix data into newstorage.
 * @param [in] data a matrix [n_data, n_dims] containing the dataset.
 * @param [in] dimOut the dimension index of newstorage where part of data will be stored.
 * @param [in] dimIn the dimension index from data to be stored into newstorage.
 * @param [in] start the first element of the peer.
 * @param [in] end the last element of the peer.
 * @param [in,out] newstorage a matrix to store the information copied from data.
 * @return 0 if it is correct, -2 if data or newstorage are NULL.
 */
int copyDimension(double **data, int dimOut, int dimIn, long start, long end, double **newstorage);
/**
 * This function computes the covariance matrix for a single peer.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] covarMatrixDim the number of dimensions of the covariance matrix.
 * @param [in] space a matrix [covarMatrixDim, partitionSize] containing the data
 *                      on which the covariance matrix is computed.
 * @param [in, out] covarianceMatrix a matrix [covarMatrixDim, covarMatrixDim] to
 *                                      store the covariance matrix.
 * @return 0 if it is correct, -2 if space or covarianceMatrix are NULL.
 */
int computeLocalCovarianceMatrix(long partitionSize, int covarMatrixDim, double **space, double **covarianceMatrix);
/**
 * This function computes the Principal Component Analysis for a single peer taking the
 * 2 most important principal components (with Armadillo library functions).
 * @param [in] covarianceMatrix a matrix [n_dims,n_dims] containing the covariance matrix.
 * @param [in] oldSpace a matrix [n_dims,partitionSize] containing the space on which the
 *                  covarianceMatrix was computed.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] n_dims the number of dimensions.
 * @param [in,out] newSpace a matrix [2, partitionSize] to store the resulting space.
 * @return 0 if it is correct, -2 if covarianceMatrix, oldSpace or newSpace are NULL.
 */
int computePCA(double **covarianceMatrix, double **oldSpace, long partitionSize, int n_dims, double **newSpace);
/**
 * This function runs the K-Means on the part of the dataset owned by a single peer. It
 * computes the local sum of the points in each cluster, the cardinality of each cluster (weights)
 * and the associated error.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] centroids a mat structure (from Armadillo).
 * @param [in] subspace a matrix [2, partitionSize] containing the data on which
 *                      executes clustering.
 * @param [in,out] weights an array [1, nCluster] containing the cardinality of the clusters.
 * @param [in,out] localsum a matrix [2, nCluster] containing the local sum of the points in each cluster.
 * @param [in,out] error a double value containing the overall error among centroids and points in the
 *                          associated cluster.
 * @return 0 if it is correct, -2 if subspace, weights, localsum are NULL or centroids is empty;
 */
int computeLocalKMeans(long partitionSize, mat centroids, double **subspace, double *weights, double **localsum, double &error);
/**
 * This function computes the mean on each of the 2 dimensions in data for a single peer.
 * @param [in] data a matrix [2, partitionSize] containing the data partition of a peer.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in, out] summaries an array [1, 2] to store the mean.
 * @return 0 if it is correct, -2 if data or summaries are NULL.
 */
int computeLocalC_Mean(double **data, long partitionSize, double *summaries);
/**
 * This function computes the number of elements in each cluster for a single peer.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in, out] pts_incluster an array [1, ncluster] to store the computed values.
 * @param [in] rep a cluster_report structure containing the clustering information.
 * @return 0 if it is correct, -2 if pts_incluster or rep.cidx are NULL.
 */
int computeLocalPtsIncluster(long partitionSize, double *pts_incluster, cluster_report rep);
/**
 * This function computes the number of inter-cluster (Nout) and
 * intra-cluster(Nin) pairs.
 * @param [in] nCluster the number of clusters.
 * @param [in] pts_incluster an array [1, ncluster] containing the number of elements
 *                              in each cluster.
 * @param [in,out] Nin a double value storing the number of intra-cluster pairs.
 * @param [in,out] Nout a double value storing the number of inter-cluster pairs.
 * @return 0 if it is correct, -2 if pts_incluster is NULL.
 */
int computeNin_Nout(int nCluster, double *pts_incluster, double &Nin, double &Nout);
/**
 * This function computes the overall intra-cluster distance.
 * The program exits with code -2 if pts_incluster or c_mean are NULL or
 * centroids is empty.
 * @param [in] pts_incluster an array [1, ncluster] containing the number of elements
 *                              in each cluster.
 * @param [in] centroids a mat structure (from Armadillo).
 * @param [in] c_mean an array containing the mean for each of the 2 dimensions.
 * @return a double value indicating the intra-cluster distance.
 */
double computeBCSS(double *pts_incluster, mat centroids, double *c_mean);
/**
 * This function computes the inter-cluster distance for a single peer.
 * The program exits with code -2 if rep.cidx or subspace are NULL or
 * rep.centroids is empty.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] rep a cluster_report structure containing the clustering information.
 * @param [in] subspace a matrix [2, partitionSize] containing the data on which
 *                  clustering was executed.
 * @return a double value indicating the inter-cluster distance.
 */
double computeLocalWCweight(long partitionSize, cluster_report rep, double **subspace);
/**
 * This function computes the distance between each point (not discarded previously)
 * managed by a single peer and its centroids.
 * The program exits with code -2 if discarded, subspace or rep.cidx are NULL or
 * rep.centroids is empty.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] rep a cluster_report structure containing the clustering information.
 * @param [in] clusterid the index indicating the considered cluster.
 * @param [in] discarded an array [1, partitionSize] indicating if a point was
 *                          discarded in a previous iteration.
 * @param [in] subspace a matrix [2, partitionSize] containing the data on which the clustering was executed.
 * @return a double value indicating the overall distance.
 */
double computeLocalClusterDistance(long partitionSize, cluster_report rep, int clusterid, bool *discarded, double **subspace);
/**
 * This function computes the number of elements (not discarded previously) managed by
 * a single peer in the given cluster. The program exits with code -2 if discarded or
 * rep.cidx are NULL.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] rep a cluster_report structure containing the clustering information.
 * @param [in] clusterid the index indicating the considered cluster.
 * @param [in] discarded an array [1, partitionSize] indicating if a point was discarded in a previous iteration.
 * @return the count of elements (not discarded previously) in the cluster.
 */
int computeLocalClusterDimension(long partitionSize, cluster_report rep, int clusterid, bool *discarded);
/**
 * This function computes the points managed by a single peer which are in the circle
 * and it updates "discarded". The program exits with code -2 if discarded, subspace or
 * rep.cidx are NULL or rep.centroids is empty.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] rep a cluster_report structure containing the clustering information.
 * @param [in] clusterid the index indicating the considered cluster.
 * @param [in,out] discarded an array [1, partitionSize] indicating if a point was discarded in a previous iteration.
 * @param [in] subspace a matrix [2, partitionSize] containing the data on which the clustering was executed.
 * @param [in] radius the radius of the circle to be used.
 * @return the count of inliers found in this iteration.
 */
int computeLocalInliers(long partitionSize, cluster_report rep, int clusterid, bool *discarded, double **subspace, double radius);
/**
 * This function computes the number of times each point managed by the peer
 * is considered as outlier in all the subspaces.
 * @param [in] uncorr_vars the number of dimensions in UNCORR.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] discarded a matrix [uncorr_vars, partitionSize] indicating if a point was discarded.
 * @param [in,out] outliers a structure [peers] storing the number of times each point is an outlier in all the subspaces.
 * @return 0 if it is correct, -2 if discarded or outliers are NULL.
 */
int getCountOutliersinSubspace(int uncorr_vars, long partitionSize, bool **discarded, vector<int> &outliers);