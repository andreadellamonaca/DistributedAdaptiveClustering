#include <iostream>
#include <igraph/igraph.h>
#include <cstring>
#include <random>
#include "adaptive_clustering.h"
#include "graph.h"
#include "error.h"

/**
 * @file main.cpp
 */

/**
 * @struct Params
 * A structure containing parameters read from command-line.
 */
struct Params {
    int          peers = 10; /**< The number of peers. */
    string       inputFilename = "../datasets/Iris.csv"; /**< The path for the input CSV file. */
    string       outputFilename; /**< The path for the output file. */
    double       convThreshold = 0.001; /**< The local convergence tolerance for the consensus algorithm. */
    int          convLimit = 3; /**< The number of consecutive rounds in which a peer must locally converge. */
    int          graphType = 2; /**< The graph distribution: 1 Geometric, 2 Barabasi-Albert, 3 Erdos-Renyi,
 *                                                          4 Regular (clique) */
    int          fanOut = 3; /**< The number of communication that a peer can carry out in a round. */
    int          roundsToExecute = -1; /**< The number of rounds to carry out in the consensus algorithm. */
    long         k_max = 10; /**< The maximum number of cluster to try for the K-Means algorithm. */
    double       elbowThreshold = 0.25; /**< The error tolerance for the selected metric to evaluate the elbow
 *                                              in K-means algorithm. */
    double       convClusteringThreshold = 0.0001; /**< The local convergence tolerance for distributed K-Means. */
    double       percentageIncircle = 0.9; /**< The percentage of points in a cluster to be considered as inliers. */
    double       percentageSubspaces = 0.8; /**< The percentage of subspace in which a point must be outlier to be
 *                                              evaluated as general outlier. */
};

chrono::high_resolution_clock::time_point t1, t2;

/**
 * This function saves the actual time into global variable t1.
 */
void StartTheClock();
/**
 * This function saves the actual time into global variable t2
 * and it computes the difference between t1 and t2.
 * @return the difference between t1 and t2
 */
double StopTheClock();
/**
 * This function handles the arguments passed by command line
 * @param [in] argv it contains command line arguments.
 * @param [in] argc it is the count of command line arguments.
 * @param [in,out] params structure to save arguments.
 * @return 0 if the command line arguments are ok, otherwise -5.
 */
int parseCommandLine(char **argv, int argc, Params &params);
/**
 * Print the needed parameters in order to run the script.
 * @param [in] cmd The name of the script.
 */
void usage(char* cmd);
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
 * This function prints the parameters used for the run.
 * @param [in] params the structure with parameters.
 */
void printUsedParameters(Params params);
/**
 * This function computes the average on each dimension for a single peer.
 * @param [in] data a matrix [n_data, n_dims] containing the dataset.
 * @param [in] ndims the number of dimensions.
 * @param [in] start the first element of the peer.
 * @param [in] end the last element of the peer.
 * @param [in,out] summaries an array [1, n_dims] to store the average of each dimension.
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
 * This function says if a dimension have to enter in CORR set computing
 * the overall Pearson coefficient (row-wise sum of the Pearson coefficients)
 * for a given dimension. The function exit with code -2 if pcc is NULL.
 * @param [in] ndims the number of dimensions.
 * @param [in] dimensionID an index between 0 and ndims indicating the given dimension.
 * @param [in] pcc the matrix [ndims, ndims] containing the Pearson coefficients.
 * @return true if the dimension have to enter in CORR, else false.
 */
bool isCorrDimension(int ndims, int dimensionID, double **pcc);
/**
 * This function computes the number of dimensions have to enter in CORR
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
 * @param [in] dimOut the dimension index on which store the part of data into newstorage.
 * @param [in] dimIn the dimension index from data to store into newstorage.
 * @param [in] start the first element of the peer.
 * @param [in] end the last element of the peer.
 * @param [in,out] newstorage a matrix to store the copy from data.
 * @return 0 if it is correct, -2 if data or newstorage are NULL.
 */
int copyDimension(double **data, int dimOut, int dimIn, long start, long end, double **newstorage);
/**
 * This function computes the covariance matrix for a single peer.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] covarMatrixDim the number of dimensions of the covariance matrix.
 * @param [in] space a matrix [covarMatrixDim, partitionSize] containing the data
 *                      on which computes the covariance matrix.
 * @param [in, out] covarianceMatrix a matrix [covarMatrixDim, covarMatrixDim] to
 *                                      store the covariance matrix.
 * @return 0 if it is correct, -2 if space or covarianceMatrix are NULL.
 */
int computeLocalCovarianceMatrix(long partitionSize, int covarMatrixDim, double **space, double **covarianceMatrix);
/**
 * This function computes the Principal Component Analysis for a single peer taking the
 * 2 principal components (with Armadillo library functions).
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
 * @param [in,out] error a double value containing the overall error between centroids and points in the
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
 * @param [in] rep a cluster_report structure containing the clustering informations.
 * @return 0 if it is correct, -2 if pts_incluster or rep.cidx are NULL.
 */
int computeLocalPtsIncluster(long partitionSize, double *pts_incluster, cluster_report rep);
/**
 * This function computes the number of inter-cluster (Nout) and
 * intra-cluster(Nin) pairs.
 * @param [in] nCluster the number of clusters.
 * @param [in] pts_incluster an array [1, ncluster] containing the number of elements
 *                              in each cluster.
 * @param [in,out] Nin a double value to store the number of intra-cluster pairs.
 * @param [in,out] Nout a double value to store the number of inter-cluster pairs.
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
 * @param [in] rep a cluster_report structure containing the clustering informations.
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
 * @param [in] rep a cluster_report structure containing the clustering informations.
 * @param [in] clusterid the index indicating the considered cluster.
 * @param [in] discarded an array [1, partitionSize] indicating if a point was
 *                          discarded in a previous iteration.
 * @param [in] subspace a matrix [2, partitionSize] containing the data on which
 *                  clustering was executed.
 * @return a double value indicating the overall distance.
 */
double computeLocalClusterDistance(long partitionSize, cluster_report rep, int clusterid, bool *discarded, double **subspace);
/**
 * This function computes the number of elements (not discarded previously) managed by
 * a single peer in the given cluster. The program exits with code -2 if discarded or
 * rep.cidx are NULL.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] rep a cluster_report structure containing the clustering informations.
 * @param [in] clusterid the index indicating the considered cluster.
 * @param [in] discarded an array [1, partitionSize] indicating if a point was
 *                          discarded in a previous iteration.
 * @return the count of elements (not discarded previously) in the cluster.
 */
int computeLocalClusterDimension(long partitionSize, cluster_report rep, int clusterid, bool *discarded);
/**
 * This function computes the points managed by a single peer which are in the circle
 * and it updates "discarded". The program exits with code -2 if discarded, subspace or
 * rep.cidx are NULL or rep.centroids is empty.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] rep a cluster_report structure containing the clustering informations.
 * @param [in] clusterid the index indicating the considered cluster.
 * @param [in,out] discarded an array [1, partitionSize] indicating if a point was
 *                          discarded in a previous iteration.
 * @param [in] subspace a matrix [2, partitionSize] containing the data on which
 *                      clustering was executed.
 * @param [in] radius the radius of the circle to be used.
 * @return the count of inliers found in this iteration.
 */
int computeLocalInliers(long partitionSize, cluster_report rep, int clusterid, bool *discarded, double **subspace, double radius);
/**
 * This function computes the number of times each point managed by the peer
 * is considered as outliers in all the subspaces.
 * @param [in] uncorr_vars the number of dimensions in UNCORR.
 * @param [in] partitionSize the number of elements managed by the peer.
 * @param [in] start_idx the index, a value between 0 and n_data, of the first
 *                          element of the peer.
 * @param [in] discarded a matrix [uncorr_vars, partitionSize] indicating if a
 *                        point was discarded.
 * @param [in,out] outliers an array [1,n_data] to store the number of times
 *                          each point is an outlier in all the subspaces.
 * @return 0 if it is correct, -2 if discarded or outliers are NULL.
 */
int getCountOutliersinSubspace(int uncorr_vars, long partitionSize, int start_idx, bool **discarded, double *outliers);
/**
 * This function computes the average between 2 values and theresult is stored
 * in the first argument.
 * @param [in,out] x the address of the first value to be averaged.
 * @param [in] y the second value to be averaged.
 * @return 0 if it is correct, -2 if x is NULL.
 */
int computeAverage(double *x, double y);
/**
 * This function merge 2 arrays with point-wise average.
 * @param [in] dim the dimension of the array.
 * @param [in,out] peerVector the first array to be merged.
 * @param [in,out] neighborVector the second array to be merged.
 * @return 0 if it is correct, -2 if peerVector or neighborVector are NULL.
 */
int mergeVector(int dim, double *peerVector, double *neighborVector);
/**
 * This function merge the upper diagonal of a square matrix with point-wise average.
 * @param [in] n_dims the dimension of the matrix.
 * @param [in,out] peer_UDiagMatrix the first matrix to be merged.
 * @param [in,out] neighbor_UDiagMatrix the second matrix to be merged.
 * @return 0 if it is correct, -2 if peer_UDiagMatrix or neighbor_UDiagMatrix are NULL.
 */
int mergeUDiagMatrix(int n_dims, double **peer_UDiagMatrix, double **neighbor_UDiagMatrix);
/**
 * This function computes the average consensus on a single value owned by the peers.
 * This function exits with code -2 if structure is NULL, -1 for error in memory allocation
 * and -10 for error on merge procedure.
 * @param [in] params the structure containing the parameters for consensus.
 * @param [in] graph an igraph_vector_t structure (from igraph).
 * @param [in] structure an array [1, peers] containing the information to be exchanged.
 * @return an array [1, peers] containing the estimate for each peer of the total number
 *          of peers (to be used for estimation of the sum of the value in structure).
 */
double* SingleValueAverageConsensus(Params params, igraph_t graph, double* structure);
/**
 * This function computes the average consensus on a vector of values owned by the peers.
 * This function exits with code -2 if structure is NULL, -1 for error in memory allocation
 * and -10 for error on merge procedure.
 * @param [in] params the structure containing the parameters for consensus.
 * @param [in] graph an igraph_vector_t structure (from igraph).
 * @param [in] structure a matrix [peers, dim] containing the information to be exchanged.
 * @return an array [1, peers] containing the estimate for each peer of the total number
 *          of peers (to be used for estimation of the sum of the value in structure).
 */
double* VectorAverageConsensus(Params params, igraph_t graph, int dim, double** structure);
/**
 * This function computes the average consensus on the upper diagonal of a square matrix
 * owned by the peers. This function exits with code -2 if structure is NULL, -1 for error
 * in memory allocation and -10 for error on merge procedure.
 * @param [in] params the structure containing the parameters for consensus.
 * @param [in] graph an igraph_vector_t structure (from igraph).
 * @param [in] structure a cube [peers, dim, dim] containing the information to be exchanged.
 * @return an array [1, peers] containing the estimate for each peer of the total number
 *          of peers (to be used for estimation of the sum of the value in structure).
 */
double* UDiagMatrixAverageConsensus(Params params, igraph_t graph, int *dim, double*** structure);
/**
 * This function computes the average consensus on localsum (clustering information)
 * owned by the peers. This function exits with code -2 if structure is NULL, -1 for error
 * in memory allocation and -10 for error on merge procedure.
 * @param [in] params the structure containing the parameters for consensus.
 * @param [in] graph an igraph_vector_t structure (from igraph).
 * @param [in] structure a cube [peers, 2, nCluster] containing the information to be exchanged.
 * @return an array [1, peers] containing the estimate for each peer of the total number
 *          of peers (to be used for estimation of the sum of the value in structure).
 */
double* LocalSumAverageConsensus(Params params, igraph_t graph, int nCluster, double*** structure);
/**
 * This function computes the average consensus on centroids owned by the peers.
 * This function exits with code -2 if structure is empty and -1 for error
 * in memory allocation.
 * @param [in] params the structure containing the parameters for consensus.
 * @param [in] graph an igraph_vector_t structure (from igraph).
 * @param [in] structure a cube structure (from Armadillo library) [peers, 2, nCluster]
 *                          containing the information to be exchanged.
 * @return an array [1, peers] containing the estimate for each peer of the total number
 *          of peers (to be used for estimation of the sum of the value in structure).
 */
double* CentroidsAverageConsensus(Params params, igraph_t graph, cube &structure);

int main(int argc, char **argv) {

    int n_dims; // number of dimensions
    int n_data; // number of data
    long *peerLastItem = NULL; // index of a peer last item
    long *partitionSize = NULL; // size of a peer partition
    Params          params;
    bool            outputOnFile;
    igraph_t        graph;
    int             programStatus = -12;
    double          elapsed;

    programStatus = parseCommandLine(argv, argc, params);
    if (programStatus) {
        return programStatus;
    }
    outputOnFile = params.outputFilename.size() > 0;
    if (!outputOnFile) {
        printUsedParameters(params);
    }

    //Structures used for consensus or convergence procedures
    double *dimestimate = nullptr;
    bool *converged = nullptr;
    int Numberofconverged;
    //Structures used for dataset loading and standardization
    double *data_storage = nullptr, **data = nullptr, *avg_storage = nullptr, **avgsummaries = nullptr;
    //Structures used for pcc and covariance computation
    double *pcc_storage = nullptr, **pcc_i = nullptr, ***pcc = nullptr, **squaresum_dims = nullptr,
            *squaresum_dims_storage = nullptr, *covar_storage = nullptr, **covar_i = nullptr, ***covar = nullptr;
    int *num_dims = nullptr;
    //Structures used for Partitioning, PCA and Subspaces
    double ***combine = nullptr, ***corr = nullptr, ****subspace = nullptr;
    int *uncorr_vars = nullptr, *corr_vars = nullptr, **uncorr = nullptr;
    //Structures used for clustering
    double *localsum_storage = nullptr, **localsum_i = nullptr, ***localsum = nullptr, *weights_storage = nullptr,
            **weights = nullptr, *prev_err = nullptr, *error = nullptr;
    cluster_report *final_i = nullptr, **final = nullptr, *prev = nullptr;
    //Structures used for BetaCV
    double *pts_incluster_storage = nullptr, **pts_incluster = nullptr, *c_mean_storage = nullptr, **c_mean = nullptr,
            *BC_weight = nullptr, *WC_weight = nullptr, *Nin_edges = nullptr, *Nout_edges = nullptr;
    //Structures used for outlier identification
    double *inliers = nullptr, *prev_inliers = nullptr, *cluster_dim = nullptr, *actual_dist = nullptr,
            *actual_cluster_dim = nullptr, *tot_num_data = nullptr, **global_outliers = nullptr;
    bool ***discardedPts = nullptr;

    /*** Dataset Loading ***/
    programStatus = getDatasetDims(params.inputFilename, n_dims, n_data);
    if (programStatus) {
        return programStatus;
    }

    data_storage = (double *) malloc(n_dims * n_data * sizeof(double));
    if (!data_storage) {
        programStatus = MemoryError(__FUNCTION__);
        return programStatus;
    }
    data = (double **) malloc(n_data * sizeof(double *));
    if (!data) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int i = 0; i < n_data; ++i) {
        data[i] = &data_storage[i * n_dims];
    }

    programStatus = loadData(params.inputFilename, data, n_dims);
    if (programStatus) {
        programStatus = DatasetReadingError(__FUNCTION__);
        goto ON_EXIT;
    }

    /*** Partitioning phase ***/
    programStatus = partitionData(n_data, params.peers, &peerLastItem, &partitionSize);
    if (programStatus) {
        goto ON_EXIT;
    }

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    /*** Graph generation ***/
    StartTheClock();

    graph = generateRandomGraph(params.graphType, params.peers);
//    igraph_vector_t result;
//    result = getMinMaxVertexDeg(graph, outputOnFile);
//    igraph_vector_destroy(&result);

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to generate the graph: " << elapsed << endl;
        cout << endl << "Applying Dataset Standardization to each peer' substream..." << endl;
    }

    /***    Dataset Standardization
     * The dataset is centered around the mean value.
     * 1) Each peer computes the local average, for each dimension, on its dataset
     * partition and stores the values on "avgsummaries".
     * 2) An average consensus is executed on the average value for each dimension.
     * 3) Each peer centers its dataset partition on the average reached with consensus.
    ***/
    StartTheClock();
    avg_storage = (double *) calloc(params.peers * n_dims, sizeof(double));
    if (!avg_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    avgsummaries = (double **) malloc(params.peers * sizeof(double *));
    if (!avgsummaries) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        avgsummaries[peerID] = &avg_storage[peerID * n_dims];

        if (peerID == 0) {
            programStatus = computeLocalAverage(data, n_dims, 0, peerLastItem[peerID], avgsummaries[peerID]);
        } else {
            programStatus = computeLocalAverage(data, n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID],
                                                avgsummaries[peerID]);
        }
        if (programStatus) {
            goto ON_EXIT;
        }
    }

    VectorAverageConsensus(params, graph, n_dims, avgsummaries);

    for(int peerID = 0; peerID < params.peers; peerID++){
        if (peerID == 0) {
            programStatus = CenterData(avgsummaries[peerID], n_dims, 0, peerLastItem[peerID], data);
        } else {
            programStatus = CenterData(avgsummaries[peerID], n_dims, peerLastItem[peerID-1] + 1,
                                       peerLastItem[peerID], data);
        }
        if (programStatus) {
            goto ON_EXIT;
        }
    }
    free(avgsummaries), avgsummaries = nullptr;
    free(avg_storage), avg_storage = nullptr;

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to standardize the dataset: " << elapsed << endl;
        cout << endl << "Computing Pearson matrix globally..." << endl;
    }

    /***    Pearson Matrix Computation
     * The Pearson Coefficient between two dimensions x and y when
     * the dataset dimensions are centered around the mean values is:
     * r_xy = sum_i of (x_i * y_i) / sqrt(sum_i of pow2(x_i) * sum_i of pow2(y_i))
     * 1) Locally each peer computes the numerator (on pcc structure) and
     *      the denominator (on squaresum_dims) of the previous r_xy for
     *      each pair of dimensions.
     * 2) A consensus on the sum of pcc and squaresum_dims is executed.
     * 3) Each peer computes the Pearson matrix with the values resulting from consensus.
    ***/
    StartTheClock();

    pcc_storage = (double *) calloc(params.peers * n_dims * n_dims, sizeof(double));
    if (!pcc_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    pcc_i = (double **) malloc(params.peers * n_dims * sizeof(double *));
    if (!pcc_i) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    pcc = (double ***) malloc(params.peers * sizeof(double **));
    if (!pcc) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    squaresum_dims_storage = (double *) calloc(params.peers * n_dims, sizeof(double));
    if (!squaresum_dims_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    squaresum_dims = (double **) malloc(params.peers * sizeof(double *));
    if (!squaresum_dims) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int i = 0; i < params.peers * n_dims; ++i) {
        pcc_i[i] = &pcc_storage[i * n_dims];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        pcc[peerID] = &pcc_i[peerID * n_dims];
        squaresum_dims[peerID] = &squaresum_dims_storage[peerID * n_dims];

        if (peerID == 0) {
            programStatus = computeLocalPCC(pcc[peerID], squaresum_dims[peerID], n_dims, 0,
                                            peerLastItem[peerID], data);
        } else {
            programStatus = computeLocalPCC(pcc[peerID], squaresum_dims[peerID], n_dims,
                                            peerLastItem[peerID-1] + 1, peerLastItem[peerID], data);
        }
        if (programStatus) {
            goto ON_EXIT;
        }
    }

    VectorAverageConsensus(params, graph, n_dims, squaresum_dims);

    num_dims = (int *) malloc(params.peers * sizeof(int));
    if (!num_dims) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(num_dims, params.peers, n_dims);
    UDiagMatrixAverageConsensus(params, graph, num_dims, pcc);
    free(num_dims), num_dims = nullptr;

    for(int peerID = 0; peerID < params.peers; peerID++){
        programStatus = computePearsonMatrix(pcc[peerID], squaresum_dims[peerID], n_dims);
        if (programStatus) {
            goto ON_EXIT;
        }
    }
    free(squaresum_dims), squaresum_dims = nullptr;
    free(squaresum_dims_storage), squaresum_dims_storage = nullptr;

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to compute the Pearson matrix: " << elapsed << endl;
        cout << endl << "Partitioning dimensions in CORR and UNCORR sets..." << endl;
    }

    cout << "PERAROAOIFNSIU"<< endl;
    for (int m = 0; m < n_dims; ++m) {
        for (int k = 0; k < n_dims; ++k) {
            cout << pcc[0][m][k] << " ";
        }
        cout << endl;
    }

    /***    CORR And UNCORR Partitioning
     * Locally each peer partitions the dimensions in CORR and UNCORR sets
     * based on the row-wise sum of the Pearson coefficient for each dimension.
     * The structures corr_vars and uncorr_vars keep records of the
     * cardinality of CORR and UNCORR sets, while "corr" contains the peers'
     * dataset partition of a dimension and "uncorr" contains the indexes of
     * dimensions with Pearson coefficient less than 0.
    ***/
    StartTheClock();

    corr_vars = (int *) calloc(params.peers, sizeof(int));
    if (!corr_vars) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    uncorr_vars = (int *) calloc(params.peers, sizeof(int));
    if (!uncorr_vars) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    uncorr = (int **) malloc(params.peers * sizeof(int *));
    if (!uncorr) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    corr = (double ***) malloc(params.peers * sizeof(double **));
    if (!corr) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        programStatus = computeCorrUncorrCardinality(pcc[peerID], n_dims, corr_vars[peerID], uncorr_vars[peerID]);
        if (programStatus) {
            goto ON_EXIT;
        }

        if (peerID == 0) {
            cout << "Correlated dimensions: " << corr_vars[peerID] << ", " << "Uncorrelated dimensions: "
                 << uncorr_vars[peerID] << endl;
            if (corr_vars[peerID] < 2) {
                programStatus = LessCorrVariablesError(__FUNCTION__);
                goto ON_EXIT;
            }
            if (uncorr_vars[peerID] == 0) {
                programStatus = NoUncorrVariablesError(__FUNCTION__);
                goto ON_EXIT;
            }
        }

        corr[peerID] = (double **) malloc(corr_vars[peerID] * sizeof(double *));
        if (!corr[peerID]) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        for (int corrVarID = 0; corrVarID < corr_vars[peerID]; ++corrVarID) {
            corr[peerID][corrVarID] = (double *) malloc(partitionSize[peerID] * sizeof(double));
            if (!corr[peerID][corrVarID]) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
        }

        uncorr[peerID] = (int *) malloc(uncorr_vars[peerID] * sizeof(int));
        if (!uncorr[peerID]) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }

        corr_vars[peerID] = 0, uncorr_vars[peerID] = 0;
        for (int dimensionID = 0; dimensionID < n_dims; ++dimensionID) {
            if (isCorrDimension(n_dims, dimensionID, pcc[peerID])) {
                if (peerID == 0) {
                    programStatus = copyDimension(data, corr_vars[peerID], dimensionID, 0,
                                                  peerLastItem[peerID], corr[peerID]);
                } else {
                    programStatus = copyDimension(data, corr_vars[peerID], dimensionID,
                                                  peerLastItem[peerID - 1] + 1, peerLastItem[peerID], corr[peerID]);
                }
                if (programStatus) {
                    goto ON_EXIT;
                }
                corr_vars[peerID]++;
            } else {
                uncorr[peerID][uncorr_vars[peerID]] = dimensionID;
                uncorr_vars[peerID]++;
            }
        }
    }
    free(pcc), pcc = nullptr;
    free(pcc_i), pcc_i = nullptr;
    free(pcc_storage), pcc_storage = nullptr;

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to partition the dimensions in CORR and UNCORR: " << elapsed << endl;
        cout << endl << "Computing Principal Component Analysis on CORR set..." << endl;
    }

    /***    Principal Component Analysis on CORR set
     * 1) Locally each peer computes the covariance matrix on its dataset
     *      partition of CORR set and saves the result in "covar".
     * 2) An average consensus on the covariance matrix is executed.
     * 3) Locally each peer computes eigenvalues/eigenvectors (with Armadillo
     *      functions), get the 2 Principal Components and store them in "combine".
    ***/
    StartTheClock();

    covar_storage = (double *) malloc(params.peers * corr_vars[0] * corr_vars[0] * sizeof(double));
    if (!covar_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    covar_i = (double **) malloc(params.peers * corr_vars[0] * sizeof(double *));
    if (!covar_i) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    covar = (double ***) malloc(params.peers * sizeof(double **));
    if (!covar) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int i = 0; i < params.peers * corr_vars[0]; ++i) {
        covar_i[i] = &covar_storage[i * corr_vars[0]];
    }
    for (int i = 0; i < params.peers; ++i) {
        covar[i] = &covar_i[i * corr_vars[0]];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        programStatus = computeLocalCovarianceMatrix(partitionSize[peerID], corr_vars[peerID],
                                                     corr[peerID], covar[peerID]);
        if (programStatus) {
            goto ON_EXIT;
        }
    }

    UDiagMatrixAverageConsensus(params, graph, corr_vars, covar);

    combine = (double ***) malloc(params.peers * sizeof(double **));
    if (!combine) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        combine[peerID] = (double **) malloc(3 * sizeof(double *));
        if (!combine[peerID]) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        for (int i = 0; i < 3; ++i) {
            combine[peerID][i] = (double *) malloc(partitionSize[peerID] * sizeof(double));
            if (!combine[peerID][i]) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
        }

        programStatus = computePCA(covar[peerID], corr[peerID], partitionSize[peerID], corr_vars[peerID],
                                   combine[peerID]);
        if (programStatus) {
            goto ON_EXIT;
        }

        for(int j = 0; j < corr_vars[peerID]; j++){
            free(corr[peerID][j]), corr[peerID][j] = nullptr;
        }
        free(corr[peerID]), corr[peerID] = nullptr;
    }
    free(corr), corr = nullptr;
    free(covar), covar = nullptr;
    free(covar_i), covar_i = nullptr;
    free(covar_storage), covar_storage = nullptr;
    free(corr_vars), corr_vars = nullptr;

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to apply PCA on CORR set: " << elapsed << endl;
        cout << endl << "Computing Candidate Subspaces through Principal Component Analysis"
                " on PC1corr, PC2corr and each dimension in the UNCORR set..." << endl;
    }

    /***    Candidate Subspaces Creation
     * 1) Locally each peer copies its dataset partition of the m-th dimension in
     * UNCORR set and computes the covariance matrix.
     * 2) An average consensus on the covariance matrix is executed.
     * 3) Locally each peer computes eigenvalues/eigenvectors (with Armadillo functions),
     * get the 2 Principal Components and store them in "subspace".
     * These operations are done for each dimension in UNCORR set.
    ***/
    StartTheClock();

    subspace = (double ****) malloc(params.peers * sizeof(double ***));
    if (!subspace) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        subspace[peerID] = (double ***) malloc(uncorr_vars[peerID] * sizeof(double **));
        if (!subspace[peerID]) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        for (int uncorrVarID = 0; uncorrVarID < uncorr_vars[peerID]; ++uncorrVarID) {
            subspace[peerID][uncorrVarID] = (double **) malloc(2 * sizeof(double *));
            if (!subspace[peerID][uncorrVarID]) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            for (int dimID = 0; dimID < 2; ++dimID) {
                subspace[peerID][uncorrVarID][dimID] = (double *) malloc(partitionSize[peerID] * sizeof(double));
                if (!subspace[peerID][uncorrVarID][dimID]) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
            }
        }
    }

    for (int subspaceID = 0; subspaceID < uncorr_vars[0]; ++subspaceID) {
        covar_storage = (double *) malloc(params.peers * 3 * 3 * sizeof(double));
        if (!covar_storage) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        covar_i = (double **) malloc(params.peers * 3 * sizeof(double *));
        if (!covar_i) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        covar = (double ***) malloc(params.peers * sizeof(double **));
        if (!covar) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        for (int i = 0; i < params.peers * 3; ++i) {
            covar_i[i] = &covar_storage[i * 3];
        }

        for(int peerID = 0; peerID < params.peers; peerID++) {
            covar[peerID] = &covar_i[peerID * 3];

            for (int i = 0; i < partitionSize[peerID]; ++i) {
                if (peerID == 0) {
                    programStatus = copyDimension(data, 2, uncorr[peerID][subspaceID], 0,
                                                  peerLastItem[peerID], combine[peerID]);
                } else {
                    programStatus = copyDimension(data, 2, uncorr[peerID][subspaceID],
                                                  peerLastItem[peerID - 1] + 1, peerLastItem[peerID], combine[peerID]);
                }
                if (programStatus) {
                    goto ON_EXIT;
                }
            }
            programStatus = computeLocalCovarianceMatrix(partitionSize[peerID], 3, combine[peerID],
                                                         covar[peerID]);
            if (programStatus) {
                goto ON_EXIT;
            }
        }

        num_dims = (int *) malloc(params.peers * sizeof(int));
        if (!num_dims) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        fill_n(num_dims, params.peers, 3);
        UDiagMatrixAverageConsensus(params, graph, num_dims, covar);
        free(num_dims), num_dims = nullptr;

        for(int peerID = 0; peerID < params.peers; peerID++) {
            programStatus = computePCA(covar[peerID], combine[peerID], partitionSize[peerID], 3,
                                       subspace[peerID][subspaceID]);
            if (programStatus) {
                goto ON_EXIT;
            }
        }
        free(covar), covar = nullptr;
        free(covar_i), covar_i = nullptr;
        free(covar_storage), covar_storage = nullptr;
    }
    for(int i = 0; i < params.peers; i++){
        free(uncorr[i]), uncorr[i] = nullptr;
        for(int j = 0; j < 3; j++){
            free(combine[i][j]), combine[i][j] = nullptr;
        }
        free(combine[i]), combine[i] = nullptr;
    }
    free(uncorr), uncorr = nullptr;
    if(combine != nullptr)
        free(combine), combine = nullptr;

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to create the candidate subspaces: " << elapsed << endl;
        cout << endl << "Computing distributed clustering..." << endl;
    }

    /***    Distributed K-Means
     * For each candidate subspace, peers cooperate to pick the optimal number of
     * clusters with the elbow criterion based on BetaCV metric. The result of the
     * clustering is stored in the structure "final".
     * 1) Peer 0 chooses a set of centroids randomly and broadcast them (through average consensus).
     * 2) All the peers start the distributed K-Means: Locally they compute the sum
     *      of all the coordinates in each cluster, the cardinality of each cluster
     *      coordinates' set and the local sum of squared errors. Then they store these
     *      informations in "localsum", "weights" and "error".
     * 3) An average consensus on the sum of localsum, weights and error is executed.
     * 4) The peers compute the new centroids and if the condition about the actual
     *      error and the previous error is satisfied, they stop the distributed
     *      K-Means with the number of cluster chosen for this run.
     * 5) They start the evaluation of BetaCV metric for the elbow criterion. If the condition
     *      about the elbow is satisfied, the distributed clustering for the actual candidate subspace is stopped.
    ***/
    StartTheClock();

    final_i = (cluster_report *) calloc(uncorr_vars[0] * params.peers, sizeof(cluster_report));
    if (!final_i) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    final = (cluster_report **) calloc(uncorr_vars[0], sizeof(cluster_report*));
    if (!final) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    for (int subspaceID = 0; subspaceID < uncorr_vars[0]; ++subspaceID) {
        final[subspaceID] = &final_i[subspaceID * params.peers];
        prev = (cluster_report *) calloc(params.peers, sizeof(cluster_report));
        if (!prev) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }

        for (int nCluster = 1; nCluster <= params.k_max; ++nCluster) {
            cube centroids(2, nCluster, params.peers, fill::zeros);
            // Peer 0 set random centroids
            centroids.slice(0) = randu<mat>(2, nCluster);
            // Broadcast random centroids
            dimestimate = CentroidsAverageConsensus(params, graph, centroids);

            for(int peerID = 0; peerID < params.peers; peerID++){
                centroids.slice(peerID) = centroids.slice(peerID) / dimestimate[peerID];
            }

            localsum_storage = (double *) malloc(nCluster * 2 * params.peers * sizeof(double));
            if (!localsum_storage) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            localsum_i = (double **) malloc(2 * params.peers * sizeof(double *));
            if (!localsum_i) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            localsum = (double ***) malloc(params.peers * sizeof(double **));
            if (!localsum) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            weights_storage = (double *) malloc(params.peers * nCluster * sizeof(double));
            if (!weights_storage) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            weights = (double **) malloc(params.peers * sizeof(double *));
            if (!weights) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            for (int i = 0; i < 2 * params.peers; ++i) {
                localsum_i[i] = &localsum_storage[i * nCluster];
            }
            for (int i = 0; i < params.peers; ++i) {
                localsum[i] = &localsum_i[i * 2];
                weights[i] = &weights_storage[i * nCluster];
            }

            prev_err = (double *) malloc(params.peers * sizeof(double));
            if (!prev_err) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            error = (double *) malloc(params.peers * sizeof(double));
            if (!error) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            fill_n(error, params.peers, 1e9);

            // Reset parameters for convergence estimate
            Numberofconverged = params.peers;
            fill_n(converged, params.peers, false);

            while( Numberofconverged ) {
                memcpy(prev_err, error, params.peers * sizeof(double));
                fill_n(weights_storage, nCluster * params.peers, 0.0);
                fill_n(localsum_storage, 2 * nCluster * params.peers, 0.0);
                fill_n(error, params.peers, 0.0);
                for(int peerID = 0; peerID < params.peers; peerID++){
                    // check peer convergence
                    if(converged[peerID])
                        continue;

                    programStatus = computeLocalKMeans(partitionSize[peerID], centroids.slice(peerID),
                                                       subspace[peerID][subspaceID], weights[peerID], localsum[peerID], error[peerID]);
                    if (programStatus) {
                        goto ON_EXIT;
                    }
                }

                LocalSumAverageConsensus(params, graph, nCluster, localsum);
                VectorAverageConsensus(params, graph, nCluster, weights);
                dimestimate = SingleValueAverageConsensus(params, graph, error);

                for(int peerID = 0; peerID < params.peers; peerID++){
                    for (int clusterID = 0; clusterID < nCluster; ++clusterID) {
                        centroids(0, clusterID, peerID) = localsum[peerID][0][clusterID] / weights[peerID][clusterID];
                        centroids(1, clusterID, peerID) = localsum[peerID][1][clusterID] / weights[peerID][clusterID];
                    }
                    error[peerID] = error[peerID] / dimestimate[peerID];
                }

                // check local convergence
                for(int peerID = 0; peerID < params.peers; peerID++){
                    if(converged[peerID])
                        continue;

                    converged[peerID] = ((prev_err[peerID] - error[peerID]) / prev_err[peerID] <= params.convClusteringThreshold);

                    if(converged[peerID]){
                        Numberofconverged --;
                    }
                }
                //cerr << "\r Active peers: " << Numberofconverged << "          ";
            }
            free(localsum), localsum = nullptr;
            free(localsum_i), localsum_i = nullptr;
            free(localsum_storage), localsum_storage = nullptr;
            free(weights), weights = nullptr;
            free(weights_storage), weights_storage = nullptr;
            free(prev_err), prev_err = nullptr;
            free(error), error = nullptr;

            for(int peerID = 0; peerID < params.peers; peerID++){
                final[subspaceID][peerID].centroids = centroids.slice(peerID);
                final[subspaceID][peerID].k = nCluster;
                if (nCluster == 1) {
                    final[subspaceID][peerID].BetaCV = 0.0;
                } else {
                    programStatus = create_cidx_matrix(subspace[peerID][subspaceID], partitionSize[peerID],
                                                       final[subspaceID][peerID]);
                    if (programStatus) {
                        goto ON_EXIT;
                    }
                }
            }
            /***    BetaCV Metric Computation
             * The peers need the intra-cluster (WC) and the inter-cluster (BC) weights,
             * the number of distinct intra-cluster (N_in) and inter-cluster (N_out) edges
             * to compute BetaCV.  For N_in and N_out, the peers get the number of elements
             * in each cluster with an average consensus on the sum of "pts_incluster".
             * For BC, the peers get the mean of all the points  with an average consensus 
             * on "c_mean". Locally all the peers compute N_in, N_out, BC and the local estimate
             * of WC; then the general WC is reached with an average consensus on the sum of WC.
             * After that each peer computes the BetaCV metric locally and evaluates
             * the elbow criterion condition.
            ***/
            if (nCluster > 1) {
                pts_incluster_storage = (double *) calloc(params.peers * nCluster, sizeof(double));
                if (!pts_incluster_storage) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
                pts_incluster = (double **) malloc(params.peers * sizeof(double *));
                if (!pts_incluster) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
                c_mean_storage = (double *) calloc(params.peers * 2, sizeof(double));
                if (!c_mean_storage) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
                c_mean = (double **) malloc(params.peers * sizeof(double *));
                if (!c_mean) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    pts_incluster[peerID] = &pts_incluster_storage[peerID * nCluster];
                    c_mean[peerID] = &c_mean_storage[peerID * 2];

                    programStatus = computeLocalC_Mean(subspace[peerID][subspaceID], partitionSize[peerID],
                                                       c_mean[peerID]);
                    if (programStatus) {
                        goto ON_EXIT;
                    }
                    programStatus = computeLocalPtsIncluster(partitionSize[peerID], pts_incluster[peerID],
                                                             final[subspaceID][peerID]);
                    if (programStatus) {
                        goto ON_EXIT;
                    }
                }

                VectorAverageConsensus(params, graph, 2, c_mean);
                dimestimate = VectorAverageConsensus(params, graph, nCluster, pts_incluster);

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    for (int clusterID = 0; clusterID < nCluster; ++clusterID) {
                        pts_incluster[peerID][clusterID] = pts_incluster[peerID][clusterID] / dimestimate[peerID];
                    }
                }

                BC_weight = (double *) calloc(params.peers, sizeof(double));
                if (!BC_weight) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
                WC_weight = (double *) calloc(params.peers, sizeof(double));
                if (!WC_weight) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
                Nin_edges = (double *) calloc(params.peers, sizeof(double));
                if (!Nin_edges) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
                Nout_edges = (double *) calloc(params.peers, sizeof(double));
                if (!Nout_edges) {
                    programStatus = MemoryError(__FUNCTION__);
                    goto ON_EXIT;
                }
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    programStatus = computeNin_Nout(nCluster, pts_incluster[peerID], Nin_edges[peerID],
                                                    Nout_edges[peerID]);
                    if (programStatus) {
                        goto ON_EXIT;
                    }
                    BC_weight[peerID] = computeBCSS(pts_incluster[peerID], final[subspaceID][peerID].centroids,
                                                    c_mean[peerID]);
                    WC_weight[peerID] = computeLocalWCweight(partitionSize[peerID], final[subspaceID][peerID],
                                                             subspace[peerID][subspaceID]);
                }
                free(pts_incluster), pts_incluster = nullptr;
                free(pts_incluster_storage), pts_incluster_storage = nullptr;
                free(c_mean), c_mean = nullptr;
                free(c_mean_storage), c_mean_storage = nullptr;

                dimestimate = SingleValueAverageConsensus(params, graph, WC_weight);

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    WC_weight[peerID] = WC_weight[peerID] / dimestimate[peerID];
                    final[subspaceID][peerID].BetaCV = (Nout_edges[peerID] * WC_weight[peerID]) / (Nin_edges[peerID] * BC_weight[peerID]);
                }
                free(BC_weight), BC_weight = nullptr;
                free(WC_weight), WC_weight = nullptr;
                free(Nin_edges), Nin_edges = nullptr;
                free(Nout_edges), Nout_edges = nullptr;

                if (fabs(prev[0].BetaCV - final[subspaceID][0].BetaCV) <= params.elbowThreshold) {
                    cout << "The optimal K is " << final[subspaceID][0].k << endl;
                    break;
                } else {
                    for (int peerID = 0; peerID < params.peers; peerID++) {
                        prev[peerID] = final[subspaceID][peerID];
                    }
                }
            }
        }
        free(prev), prev = nullptr;
    }

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to run K-Means: " << elapsed << endl;
        cout << endl << "Starting the outlier identification process..." << endl;
    }

    /***    Outliers Identification On Each Candidate Subspace
     * For each cluster:
     * 1) Each peer computes the local cluster size and saves it in "cluster_dim".
     * 2) The cardinality of the cluster is computed with an average consensus on
     *      "cluster_dim".
     * Then it starts the distributed outlier identification:
     * 1) Each peer computes the actual cluster dimension (cluster dimension without
     *      points discarded, considered as inliers in previous iterations) and stores
     *      it in "actual_cluster_dim" and the distance between the points and the centroids.
     * 2) These informations are exchanged with an average consensus and after that each peer
     *      computes the radius of the circle used to discard points considered as inliers.
     * 3) Each peer evaluates the remaining points and if there are inliers, it updates "inliers".
     * 4) This information is exchanged with an average consensus on the sum of "inliers" and if
     *      one of the conditions on the inliers is satisfied, the outlier identification for the
     *      actual cluster is stopped.
    ***/
    StartTheClock();
    // Structure to keep record of inliers for each peer (for Outlier identification)
    discardedPts = (bool ***) malloc(params.peers * sizeof(bool **));
    if (!discardedPts) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for(int peerID = 0; peerID < params.peers; peerID++) {
        discardedPts[peerID] = (bool **) malloc(uncorr_vars[peerID] * sizeof(bool *));
        if (!discardedPts[peerID]) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        for (int uncorrVarID = 0; uncorrVarID < uncorr_vars[peerID]; ++uncorrVarID) {
            discardedPts[peerID][uncorrVarID] = (bool *) calloc(partitionSize[peerID], sizeof(bool));
            if (!discardedPts[peerID][uncorrVarID]) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
        }
    }

    for (int subspaceID = 0; subspaceID < uncorr_vars[0]; ++subspaceID) {
        inliers = (double *) malloc(params.peers * sizeof(double));
        if (!inliers) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        prev_inliers = (double *) malloc(params.peers * sizeof(double));
        if (!prev_inliers) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }
        cluster_dim = (double *) malloc(params.peers * sizeof(double));
        if (!cluster_dim) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }

        for (int clusterID = 0; clusterID < final[subspaceID][0].k; ++clusterID) {
            for (int peerID = 0; peerID < params.peers; peerID++) {
                cluster_dim[peerID] = cluster_size(final[subspaceID][peerID], clusterID, partitionSize[peerID]);
            }

            SingleValueAverageConsensus(params, graph, cluster_dim);

            // Reset parameters for convergence estimate
            fill_n(inliers, params.peers, 0);
            Numberofconverged = params.peers;
            fill_n(converged, params.peers, false);
            actual_dist = (double *) calloc(params.peers, sizeof(double));
            if (!actual_dist) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }
            actual_cluster_dim = (double *) calloc(params.peers, sizeof(double));
            if (!actual_cluster_dim) {
                programStatus = MemoryError(__FUNCTION__);
                goto ON_EXIT;
            }

            while (Numberofconverged) {
                memcpy(prev_inliers, inliers, params.peers * sizeof(double));
                fill_n(actual_dist, params.peers, 0.0);
                fill_n(actual_cluster_dim, params.peers, 0.0);
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    if (converged[peerID])
                        continue;

                    actual_cluster_dim[peerID] = computeLocalClusterDimension(partitionSize[peerID],
                            final[subspaceID][peerID], clusterID, discardedPts[peerID][subspaceID]);
                    actual_dist[peerID] = computeLocalClusterDistance(partitionSize[peerID],
                            final[subspaceID][peerID], clusterID, discardedPts[peerID][subspaceID],
                                                                      subspace[peerID][subspaceID]);
                }

                SingleValueAverageConsensus(params, graph, actual_dist);
                SingleValueAverageConsensus(params, graph, actual_cluster_dim);

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    double dist_mean = actual_dist[peerID] / actual_cluster_dim[peerID];
                    inliers[peerID] += computeLocalInliers(partitionSize[peerID], final[subspaceID][peerID],
                            clusterID, discardedPts[peerID][subspaceID], subspace[peerID][subspaceID], dist_mean);
                }

                SingleValueAverageConsensus(params, graph, inliers);

                // check local convergence
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    if (converged[peerID])
                        continue;

                    converged[peerID] = ( (inliers[peerID] >= params.percentageIncircle * cluster_dim[peerID])
                                          || prev_inliers[peerID] == inliers[peerID] );
                    if (converged[peerID]) {
                        Numberofconverged--;
                    }
                }
            }
            free(actual_dist), actual_dist = nullptr;
            free(actual_cluster_dim), actual_cluster_dim = nullptr;
        }
        free(inliers), inliers = nullptr;
        free(prev_inliers), prev_inliers = nullptr;
        free(cluster_dim), cluster_dim = nullptr;
    }

    /***    Final result broadcast to all the peers
     * Each peer sets "tot_num_data" to its partition size and then all the peers
     * estimate the total number of data with an average consensus on the sum of "tot_num_data".
     * Each peer fills "global_outliers" with the local information, then the global solution is
     * reached with an average consensus on the sum of each element in "global_outliers".
     * Finally peer 0 is queried to get the final result.
    ***/
    tot_num_data = (double *) calloc(params.peers, sizeof(double));
    if (!tot_num_data) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for(int peerID = 0; peerID < params.peers; peerID++){
        tot_num_data[peerID] = partitionSize[peerID];
    }

    dimestimate = SingleValueAverageConsensus(params, graph, tot_num_data);

    global_outliers = (double **) malloc(params.peers * sizeof(double *));
    if (!global_outliers) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    for (int peerID = 0; peerID < params.peers; peerID++) {
        tot_num_data[peerID] = std::round(tot_num_data[peerID] / (double) dimestimate[peerID]);

        global_outliers[peerID] = (double *) calloc(tot_num_data[peerID], sizeof(double));
        if (!global_outliers[peerID]) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }

        if (peerID == 0) {
            programStatus = getCountOutliersinSubspace(uncorr_vars[peerID], partitionSize[peerID],
                                                       0, discardedPts[peerID], global_outliers[peerID]);
        } else {
            int index = peerLastItem[peerID - 1] + 1;
            programStatus = getCountOutliersinSubspace(uncorr_vars[peerID], partitionSize[peerID],
                                                       index, discardedPts[peerID], global_outliers[peerID]);
        }
        if (programStatus) {
            goto ON_EXIT;
        }
    }

    dimestimate = VectorAverageConsensus(params, graph, tot_num_data[0], global_outliers);

    for (int peerID = 0; peerID < params.peers; peerID++) {
        for (int dataID = 0; dataID < tot_num_data[peerID]; ++dataID) {
            global_outliers[peerID][dataID] = std::round(global_outliers[peerID][dataID] / dimestimate[peerID]);
        }
    }

    cout << endl << "OUTLIERS:" << endl;
    for (int dataID = 0; dataID < tot_num_data[0]; ++dataID) {
        if (global_outliers[0][dataID] >= std::round(uncorr_vars[0] * params.percentageSubspaces)) {
            cout << dataID << ") ";
            for (int l = 0; l < n_dims; ++l) {
                cout << data[dataID][l] << " ";
            }
            cout << "(" << global_outliers[0][dataID] << ")" << endl;
        }
    }
    //data_out(subspace, peerLastItem, "iris.csv", discardedPts, params.peers, uncorr_vars[0], final[0]);

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to identify the outliers: " << elapsed << endl;
    }

    igraph_destroy(&graph);
    programStatus = 0;

    ON_EXIT:

    if(data != nullptr)
        free(data), data = nullptr;
    if(data_storage != nullptr)
        free(data_storage), data_storage = nullptr;
    if(peerLastItem != nullptr)
        free(peerLastItem), peerLastItem = nullptr;
    if(partitionSize != nullptr)
        free(partitionSize), partitionSize = nullptr;
    if(converged != nullptr)
        free(converged), converged = nullptr;
    if(avgsummaries != nullptr)
        free(avgsummaries), avgsummaries = nullptr;
    if(avg_storage != nullptr)
        free(avg_storage), avg_storage = nullptr;
    if(pcc != nullptr)
        free(pcc), pcc = nullptr;
    if(pcc_i != nullptr)
        free(pcc_i), pcc_i = nullptr;
    if(pcc_storage != nullptr)
        free(pcc_storage), pcc_storage = nullptr;
    if(squaresum_dims != nullptr)
        free(squaresum_dims), squaresum_dims = nullptr;
    if(squaresum_dims_storage != nullptr)
        free(squaresum_dims_storage), squaresum_dims_storage = nullptr;
    if(num_dims != nullptr)
        free(num_dims), num_dims = nullptr;
    if(uncorr != nullptr) {
        for(int i = 0; i < params.peers; i++){
            if(uncorr[i] != nullptr)
                free(uncorr[i]), uncorr[i] = nullptr;
        }
        free(uncorr), uncorr = nullptr;
    }
    if(corr_vars != nullptr) {
        if(corr != nullptr)
            for(int i = 0; i < params.peers; i++){
                for(int j = 0; j < corr_vars[i]; j++){
                    if(corr[i][j] != nullptr)
                        free(corr[i][j]), corr[i][j] = nullptr;
                }
                if(corr[i] != nullptr)
                    free(corr[i]), corr[i] = nullptr;
            }
        free(corr), corr = nullptr;
        free(corr_vars), corr_vars = nullptr;
    }
    if(covar != nullptr)
        free(covar), covar = nullptr;
    if(covar_i != nullptr)
        free(covar_i), covar_i = nullptr;
    if(covar_storage != nullptr)
        free(covar_storage), covar_storage = nullptr;
    if(combine != nullptr) {
        for(int i = 0; i < params.peers; i++){
            for(int j = 0; j < 3; j++){
                if(combine[i][j] != nullptr)
                    free(combine[i][j]), combine[i][j] = nullptr;
            }
            if(combine[i] != nullptr)
                free(combine[i]), combine[i] = nullptr;
        }
        free(combine), combine = nullptr;
    }
    if(uncorr_vars != nullptr) {
        if(subspace != nullptr) {
            for(int i = 0; i < params.peers; i++){
                for(int j = 0; j < uncorr_vars[i]; j++){
                    for (int k = 0; k < 2; ++k) {
                        if(subspace[i][j][k] != nullptr)
                            free(subspace[i][j][k]), subspace[i][j][k] = nullptr;
                    }
                    if(subspace[i][j] != nullptr)
                        free(subspace[i][j]), subspace[i][j] = nullptr;
                }
                if(subspace[i] != nullptr)
                    free(subspace[i]), subspace[i] = nullptr;
            }
        }
        free(subspace), subspace = nullptr;
        if(discardedPts != nullptr) {
            for(int i = 0; i < params.peers; i++){
                for(int j = 0; j < uncorr_vars[i]; j++){
                    if(discardedPts[i][j] != nullptr)
                        free(discardedPts[i][j]), discardedPts[i][j] = nullptr;
                }
                if(discardedPts[i] != nullptr)
                    free(discardedPts[i]), discardedPts[i] = nullptr;
            }
            free(discardedPts), discardedPts = nullptr;
        }
        free(uncorr_vars), uncorr_vars = nullptr;
    }
    if(final_i != nullptr)
        free(final_i), final_i = nullptr;
    if(final != nullptr)
        free(final), final = nullptr;
    if(prev != nullptr)
        free(prev), prev = nullptr;
    if(localsum != nullptr)
        free(localsum), localsum = nullptr;
    if(localsum_i != nullptr)
        free(localsum_i), localsum_i = nullptr;
    if(localsum_storage != nullptr)
        free(localsum_storage), localsum_storage = nullptr;
    if(weights != nullptr)
        free(weights), weights = nullptr;
    if(weights_storage != nullptr)
        free(weights_storage), weights_storage = nullptr;
    if(prev_err != nullptr)
        free(prev_err), prev_err = nullptr;
    if(error != nullptr)
        free(error), error = nullptr;
    if(pts_incluster != nullptr)
        free(pts_incluster), pts_incluster = nullptr;
    if(pts_incluster_storage != nullptr)
        free(pts_incluster_storage), pts_incluster_storage = nullptr;
    if(c_mean != nullptr)
        free(c_mean), c_mean = nullptr;
    if(c_mean_storage != nullptr)
        free(c_mean_storage), c_mean_storage = nullptr;
    if(BC_weight != nullptr)
        free(BC_weight), BC_weight = nullptr;
    if(WC_weight != nullptr)
        free(WC_weight), WC_weight = nullptr;
    if(Nin_edges != nullptr)
        free(Nin_edges), Nin_edges = nullptr;
    if(Nout_edges != nullptr)
        free(Nout_edges), Nout_edges = nullptr;
    if(inliers != nullptr)
        free(inliers), inliers = nullptr;
    if(prev_inliers != nullptr)
        free(prev_inliers), prev_inliers = nullptr;
    if(cluster_dim != nullptr)
        free(cluster_dim), cluster_dim = nullptr;
    if(actual_dist != nullptr)
        free(actual_dist), actual_dist = nullptr;
    if(actual_cluster_dim != nullptr)
        free(actual_cluster_dim), actual_cluster_dim = nullptr;
    if(tot_num_data != nullptr)
        free(tot_num_data), tot_num_data = nullptr;
    if(global_outliers != nullptr) {
        for(int i = 0; i < params.peers; i++){
            if(global_outliers[i] != nullptr)
                free(global_outliers[i]), global_outliers[i] = nullptr;
        }
        free(global_outliers), global_outliers = nullptr;
    }
    if(dimestimate != nullptr)
        free(dimestimate), dimestimate = nullptr;

    return programStatus;
}

void StartTheClock(){
    t1 = chrono::high_resolution_clock::now();
}

double StopTheClock() {
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    return time_span.count();
}

int parseCommandLine(char **argv, int argc, Params &params) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-p") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of peers parameter." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.peers = stoi(argv[i]);
        } else if (strcmp(argv[i], "-f") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing fan-out parameter." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.fanOut = stol(argv[i]);
        } else if (strcmp(argv[i], "-d") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing graph type parameter." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.graphType = stoi(argv[i]);
        } else if (strcmp(argv[i], "-ct") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing convergence tolerance parameter." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.convThreshold = stod(argv[i]);
        } else if (strcmp(argv[i], "-cl") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing # of consecutive rounds in which convergence is satisfied parameter." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.convLimit = stol(argv[i]);
        } else if (strcmp(argv[i], "-of") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing filename for simulation output." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.outputFilename = string(argv[i]);
        } else if (strcmp(argv[i], "-r") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of rounds to execute." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.roundsToExecute = stoi(argv[i]);
        } else if (strcmp(argv[i], "-k") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing max number of clusters for Elbow method." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.k_max = stol(argv[i]);
        } else if (strcmp(argv[i], "-et") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing threshold for Elbow method." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.convClusteringThreshold = stof(argv[i]);
        } else if (strcmp(argv[i], "-clst") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing threshold for distributed clustering." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.elbowThreshold = stof(argv[i]);
        } else if (strcmp(argv[i], "-pi") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of inlier points." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.percentageIncircle = stof(argv[i]);
        } else if (strcmp(argv[i], "-ps") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of subspace in which an outlier must be." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.percentageSubspaces = stof(argv[i]);
        } else if (strcmp(argv[i], "-if") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing input file name." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.inputFilename = string(argv[i]);
        } else {
            usage(argv[0]);
            return ArgumentsError(__FUNCTION__);
        }
    }
    return 0;
}

void usage(char* cmd)
{
    cerr
            << "Usage: " << cmd << "\n"
            << "-p          number of peers" << endl
            << "-f          fan-out of peers" << endl
            << "-s          seed" << endl
            << "-d          graph type: 1 geometric 2 Barabasi-Albert 3 Erdos-Renyi 4 regular" << endl
            << "-ct         convergence tolerance" << endl
            << "-cl         number of consecutive rounds in which convergence must be satisfied" << endl
            << "-of         output filename, if specified a file with this name containing all of the peers stats is written" << endl
            << "-k          max number of clusters to try in elbow criterion" << endl
            << "-et         threshold for the selection of optimal number of clusters in Elbow method" << endl
            << "-clst       the local convergence tolerance for distributed K-Means" << endl
            << "-pi         percentage of points in a cluster to be evaluated as inlier" << endl
            << "-ps         percentage of subspaces in which a point must be outlier to be evaluated as general outlier" << endl
            << "-if         input filename" << endl << endl;
}

int partitionData(int n_data, int peers, long **peerLastItem, long **partitionSize) {

    *peerLastItem = (long *) calloc(peers, sizeof(long));
    if (!(*peerLastItem)) {
        return MemoryError(__FUNCTION__);
    }
    *partitionSize = (long *) calloc(peers, sizeof(long));
    if (!(*partitionSize)) {
        free(*peerLastItem), *peerLastItem = nullptr;
        return MemoryError(__FUNCTION__);
    }
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<> distr(-1, 1); // define the range

    for(int i = 0; i < peers - 1; i++){
        float rnd = distr(eng);
        //cerr << "rnd: " << rnd << "\n";
        long last_item = rnd * ((float)n_data/(float)peers) * 0.1 + (float) (i+1) * ((float)n_data/(float)peers) - 1;
        (*peerLastItem)[i] = last_item;
    }
    (*peerLastItem)[peers - 1] = n_data-1;

    /*** Check the partitioning correctness ***/
    long sum = (*peerLastItem)[0] + 1;
    (*partitionSize)[0] = (*peerLastItem)[0] + 1;
    //cerr << "peer 0:" << sum << "\n";
    for(int i = 1; i < peers; i++) {
        sum += (*peerLastItem)[i] - (*peerLastItem)[i-1];
        (*partitionSize)[i] = (*peerLastItem)[i] - (*peerLastItem)[i-1];
        //cerr << "peer " << i << ":" << (*peerLastItem)[i] - (*peerLastItem)[i-1] << "\n";
    }

    if(sum != n_data) {
        cout << "ERROR: n_data = " << n_data << "!= sum = " << sum << endl;
        return PartitioningDatasetError(__FUNCTION__);
    }
    return 0;
}

void printUsedParameters(Params params) {
    cout << endl << endl << "PARAMETERS: " << endl
         << "input file= " << params.inputFilename  << endl
         << "percentage in circle = " << params.percentageIncircle  << endl
         << "elbow threshold = " << params.elbowThreshold  << endl
         << "convergence clustering threshold = " << params.convClusteringThreshold  << endl
         << "percentage subspaces = " << params.percentageSubspaces  << endl
         << "k_max = " << params.k_max  << endl
         << "local convergence tolerance = "<< params.convThreshold  << endl
         << "number of consecutive rounds in which a peer must locally converge = "<< params.convLimit << endl
         << "peers = " << params.peers  << endl
         << "fan-out = " << params.fanOut << endl
         << "graph type = ";
    printGraphType(params.graphType);
    cout << endl << endl;

}

int computeLocalAverage(double **data, int ndims, long start, long end, double *summaries) {
    if (!data || !summaries) {
        return NullPointerError(__FUNCTION__);
    }

    double weight = 0;
    if ((end-start)) {
        weight = 1 / (double) (end-start);
    }
    for (int i = start; i <= end; ++i) {
        for (int j = 0; j < ndims; ++j) {
            summaries[j] += weight * data[i][j];
        }
    }
    return 0;
}

int CenterData(double *summaries, int ndims, long start, long end, double **data) {
    if (!summaries || !data) {
        return NullPointerError(__FUNCTION__);
    }

    for (int i = start; i <= end; ++i) {
        for (int j = 0; j < ndims; ++j) {
            data[i][j] -= summaries[j];
        }
    }
    return 0;
}

int computeLocalPCC(double **pcc, double *squaresum_dims, int ndims, long start, long end, double **data) {
    if (!pcc || !squaresum_dims || !data) {
        return NullPointerError(__FUNCTION__);
    }

    for (int l = 0; l < ndims; ++l) {
        pcc[l][l] = 1;
        for (int i = start; i <= end; ++i) {
            squaresum_dims[l] += pow(data[i][l], 2);
            for (int m = l + 1; m < ndims; ++m) {
                pcc[l][m] += data[i][l] * data[i][m];
            }
        }
    }
    return 0;
}

int computePearsonMatrix(double **pcc, double *squaresum_dims, int ndims) {
    if (!pcc || !squaresum_dims) {
        return NullPointerError(__FUNCTION__);
    }

    for (int l = 0; l < ndims; ++l) {
        for (int m = l + 1; m < ndims; ++m) {
            pcc[l][m] = pcc[l][m] / sqrt(squaresum_dims[l] * squaresum_dims[m]);
        }
    }
    return 0;
}

bool isCorrDimension(int ndims, int dimensionID, double **pcc) {
    if (!pcc) {
        exit(NullPointerError(__FUNCTION__));
    }
    double overall = 0.0;
    for (int secondDimension = 0; secondDimension < ndims; ++secondDimension) {
        if (secondDimension != dimensionID) {
            overall += pcc[dimensionID][secondDimension];
        }
    }
    return ( (overall / ndims) >= 0 );
}

int computeCorrUncorrCardinality(double **pcc, int ndims, int &corr_vars, int &uncorr_vars) {
    if (!pcc) {
        return NullPointerError(__FUNCTION__);
    }

    for (int i = 0; i < ndims; ++i) {
        if (isCorrDimension(ndims, i, pcc)) {
            corr_vars++;
        }
    }
    uncorr_vars = ndims - corr_vars;
    return 0;
}

int copyDimension(double **data, int dimOut, int dimIn, long start, long end, double **newstorage) {
    if (!data || !newstorage) {
        return NullPointerError(__FUNCTION__);
    }
    int elem = 0;
    for (int k = start; k <= end; ++k) {
        newstorage[dimOut][elem] = data[k][dimIn];
        elem++;
    }
    return 0;
}

int computeLocalCovarianceMatrix(long partitionSize, int covarMatrixDim, double **space, double **covarianceMatrix) {
    if (!space || !covarianceMatrix) {
        return NullPointerError(__FUNCTION__);
    }
    for (int i = 0; i < covarMatrixDim; ++i) {
        for (int j = i; j < covarMatrixDim; ++j) {
            covarianceMatrix[i][j] = 0;
            for (int k = 0; k < partitionSize; ++k) {
                covarianceMatrix[i][j] += space[i][k] * space[j][k];
            }
            if (partitionSize != 0) {
                covarianceMatrix[i][j] = covarianceMatrix[i][j] / partitionSize;
            }
        }
    }
    return 0;
}

int computePCA(double **covarianceMatrix, double **oldSpace, long partitionSize, int n_dims, double **newSpace) {
    if (!covarianceMatrix || !oldSpace || !newSpace) {
        return NullPointerError(__FUNCTION__);
    }
    mat cov_mat(covarianceMatrix[0], n_dims, n_dims);
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, cov_mat);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < partitionSize; ++j) {
            double value = 0.0;
            for (int k = 0; k < n_dims; ++k) {
                int col = n_dims - i - 1;
                value += oldSpace[k][j] * eigvec(k, col);
            }
            newSpace[i][j] = value;
        }
    }
    return 0;
}

int computeLocalKMeans(long partitionSize, mat centroids, double **subspace, double *weights, double **localsum, double &error) {
    if (!subspace || !weights || !localsum || centroids.is_empty()) {
        return NullPointerError(__FUNCTION__);
    }

    for (int l = 0; l < centroids.n_cols; ++l) {
        weights[l] += 1;
        localsum[0][l] += centroids(0, l);
        localsum[1][l] += centroids(1, l);
    }
    for (int k = 0; k < partitionSize; ++k) {
        int clusterid = mindistCluster(centroids, subspace[0][k], subspace[1][k]);

        weights[clusterid] += 1;
        localsum[0][clusterid] += subspace[0][k];
        localsum[1][clusterid] += subspace[1][k];
        error += pow(L2distance(centroids(0, clusterid), centroids(1, clusterid), subspace[0][k], subspace[1][k]), 2);
    }
    return 0;
}

int computeLocalC_Mean(double **data, long partitionSize, double *summaries) {
    if (!data || !summaries) {
        return NullPointerError(__FUNCTION__);
    }

    double weight = 0;
    if (partitionSize) {
        weight = 1 / (double) partitionSize;
    }
    for (int k = 0; k < partitionSize; ++k) {
        for (int l = 0; l < 2; ++l) {
            summaries[l] += weight * data[l][k];
        }
    }
    return 0;
}

int computeLocalPtsIncluster(long partitionSize, double *pts_incluster, cluster_report rep) {
    if (!pts_incluster || !(rep.cidx)) {
        return NullPointerError(__FUNCTION__);
    }

    for (int k = 0; k < partitionSize; ++k) {
        pts_incluster[rep.cidx[k]]++;
    }
    return 0;
}

int computeNin_Nout(int nCluster, double *pts_incluster, double &Nin, double &Nout) {
    if (!pts_incluster) {
        return NullPointerError(__FUNCTION__);
    }

    double nin = 0, nout = 0;
    for (int m = 0; m < nCluster; ++m) {
        nin += pts_incluster[m] * (pts_incluster[m] - 1);
        for (int k = 0; k < nCluster; ++k) {
            if (k != m) {
                nout += pts_incluster[m] * pts_incluster[k];
            }
        }
    }
    Nin = nin / 2;
    Nout = nout / 2;
    return 0;
}

double computeBCSS(double *pts_incluster, mat centroids, double *c_mean) {
    if (!pts_incluster || !c_mean || centroids.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    double bcss = 0;
    for (int m = 0; m < centroids.n_cols; ++m) {
        bcss += (pts_incluster[m] * L2distance(centroids(0, m), centroids(1, m), c_mean[0], c_mean[1]));
    }
    return bcss;
}

double computeLocalWCweight(long partitionSize, cluster_report rep, double **subspace) {
    if (!subspace || !(rep.cidx) || rep.centroids.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    double wcss = 0;
    for (int m = 0; m < rep.centroids.n_cols; ++m) {
        for (int k = 0; k < partitionSize; ++k) {
            if (rep.cidx[k] == m) {
                wcss += L2distance(rep.centroids(0, m), rep.centroids(1, m), subspace[0][k], subspace[1][k]);
            }
        }
    }
    return wcss;
}

double computeLocalClusterDistance(long partitionSize, cluster_report rep, int clusterid, bool *discarded, double **subspace) {
    if (!discarded || !subspace || !(rep.cidx) || rep.centroids.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    double dist = 0.0;
    for (int k = 0; k < partitionSize; ++k) {
        if (rep.cidx[k] == clusterid && !(discarded[k]) ) {
            dist += L2distance(rep.centroids(0, clusterid), rep.centroids(1, clusterid), subspace[0][k], subspace[1][k]);
        }
    }
    return dist;
}

int computeLocalClusterDimension(long partitionSize, cluster_report rep, int clusterid, bool *discarded) {
    if (!discarded || !(rep.cidx)) {
        exit(NullPointerError(__FUNCTION__));
    }

    int count = 0;
    for (int k = 0; k < partitionSize; ++k) {
        if (rep.cidx[k] == clusterid && !(discarded[k]) ) {
            count++;
        }
    }
    return count;
}

int computeLocalInliers(long partitionSize, cluster_report rep, int clusterid, bool *discarded, double **subspace, double radius) {
    if (!discarded || !subspace || !(rep.cidx) || rep.centroids.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    int count = 0;
    for (int k = 0; k < partitionSize; ++k) {
        if (rep.cidx[k] == clusterid && !(discarded[k]) ) {
            if (L2distance(rep.centroids(0, clusterid), rep.centroids.at(1, clusterid), subspace[0][k], subspace[1][k])
                <= radius) {
                discarded[k] = true;
                count++;
            }
        }
    }
    return count;
}

int getCountOutliersinSubspace(int uncorr_vars, long partitionSize, int start_idx, bool **discarded, double *outliers) {
    if (!discarded || !(outliers)) {
        return NullPointerError(__FUNCTION__);
    }

    for (int k = 0; k < partitionSize; ++k) {
        for (int j = 0; j < uncorr_vars; ++j) {
            if (!discarded[j][k]) {
                outliers[start_idx+k]++;
            }
        }
    }
    return 0;
}

int computeAverage(double *x, double y) {
    if (!x) {
        return NullPointerError(__FUNCTION__);
    }
    *x = (*x+y)/2.0;
    return 0;
}

int mergeVector(int dim, double *peerVector, double *neighborVector) {
    if (!peerVector || !neighborVector) {
        return NullPointerError(__FUNCTION__);
    }
    for (int j = 0; j < dim; ++j) {
        computeAverage(&peerVector[j], neighborVector[j]);
    }
    memcpy(neighborVector, peerVector, dim * sizeof(double));
    return 0;
}

int mergeUDiagMatrix(int n_dims, double **peer_UDiagMatrix, double **neighbor_UDiagMatrix) {
    if (!peer_UDiagMatrix || !neighbor_UDiagMatrix) {
        return NullPointerError(__FUNCTION__);
    }

    for (int l = 0; l < n_dims; ++l) {
        for (int m = l; m < n_dims; ++m) {
            computeAverage(&peer_UDiagMatrix[l][m], neighbor_UDiagMatrix[l][m]);
            peer_UDiagMatrix[m][l] = peer_UDiagMatrix[l][m];
        }
    }
    return 0;
}

double* SingleValueAverageConsensus(Params params, igraph_t graph, double* structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }
    double *dimestimate = nullptr, *prevestimate = nullptr;
    bool *converged = nullptr;
    int *convRounds = nullptr;
    int Numberofconverged = params.peers;
    int rounds = 0;
    int status = 0;

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(MemoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    while( (params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0){
        memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
        for(int peerID = 0; peerID < params.peers; peerID++){
            // check peer convergence
            if(params.roundsToExecute < 0 && converged[peerID])
                continue;
            // determine peer neighbors
            igraph_vector_t neighbors;
            igraph_vector_init(&neighbors, 0);
            igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
            long neighborsSize = igraph_vector_size(&neighbors);
            if(params.fanOut < neighborsSize){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                status = computeAverage(&structure[peerID], structure[neighborID]);
                if (status) {
                    exit(MergeError(__FUNCTION__));
                }
                structure[neighborID] = structure[peerID];
                computeAverage(&dimestimate[peerID], dimestimate[neighborID]);
                dimestimate[neighborID] = dimestimate[peerID];
            }
            igraph_vector_destroy(&neighbors);
        }

        // check local convergence
        if (params.roundsToExecute < 0) {
            for(int peerID = 0; peerID < params.peers; peerID++){
                if(converged[peerID])
                    continue;
                bool dimestimateconv;
                if(prevestimate[peerID])
                    dimestimateconv = fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) < params.convThreshold;
                else
                    dimestimateconv = false;

                if(dimestimateconv)
                    convRounds[peerID]++;
                else
                    convRounds[peerID] = 0;

                converged[peerID] = (convRounds[peerID] >= params.convLimit);
                if(converged[peerID]){
                    Numberofconverged --;
                }
            }
        }
        rounds++;
        //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
        params.roundsToExecute--;
    }

    ON_EXIT:

    if (converged != nullptr)
        free(converged), converged = nullptr;

    if (convRounds != nullptr)
        free(convRounds), convRounds = nullptr;

    if (prevestimate != nullptr)
        free(prevestimate), prevestimate = nullptr;

    return dimestimate;
}

double* VectorAverageConsensus(Params params, igraph_t graph, int dim, double** structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }
    double *dimestimate = nullptr, *prevestimate = nullptr;
    bool *converged = nullptr;
    int *convRounds = nullptr;
    int Numberofconverged = params.peers;
    int rounds = 0;
    int status = 0;

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(MemoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    while( (params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0){
        memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
        for(int peerID = 0; peerID < params.peers; peerID++){
            // check peer convergence
            if(params.roundsToExecute < 0 && converged[peerID])
                continue;
            // determine peer neighbors
            igraph_vector_t neighbors;
            igraph_vector_init(&neighbors, 0);
            igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
            long neighborsSize = igraph_vector_size(&neighbors);
            if(params.fanOut < neighborsSize){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                status = mergeVector(dim, structure[peerID], structure[neighborID]);
                if (status) {
                    exit(MergeError(__FUNCTION__));
                }
                computeAverage(&dimestimate[peerID], dimestimate[neighborID]);
                dimestimate[neighborID] = dimestimate[peerID];
            }
            igraph_vector_destroy(&neighbors);
        }

        // check local convergence
        if (params.roundsToExecute < 0) {
            for(int peerID = 0; peerID < params.peers; peerID++){
                if(converged[peerID])
                    continue;
                bool dimestimateconv;
                if(prevestimate[peerID])
                    dimestimateconv = fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) < params.convThreshold;
                else
                    dimestimateconv = false;

                if(dimestimateconv)
                    convRounds[peerID]++;
                else
                    convRounds[peerID] = 0;

                converged[peerID] = (convRounds[peerID] >= params.convLimit);
                if(converged[peerID]){
                    Numberofconverged --;
                }
            }
        }
        rounds++;
        //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
        params.roundsToExecute--;
    }

    ON_EXIT:

    if (converged != nullptr)
        free(converged), converged = nullptr;

    if (convRounds != nullptr)
        free(convRounds), convRounds = nullptr;

    if (prevestimate != nullptr)
        free(prevestimate), prevestimate = nullptr;

    return dimestimate;
}

double* UDiagMatrixAverageConsensus(Params params, igraph_t graph, int *dim, double*** structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }
    double *dimestimate = nullptr, *prevestimate = nullptr;
    bool *converged = nullptr;
    int *convRounds = nullptr;
    int Numberofconverged = params.peers;
    int rounds = 0;
    int status = 0;

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(MemoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    while( (params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0){
        memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
        for(int peerID = 0; peerID < params.peers; peerID++){
            // check peer convergence
            if(params.roundsToExecute < 0 && converged[peerID])
                continue;
            // determine peer neighbors
            igraph_vector_t neighbors;
            igraph_vector_init(&neighbors, 0);
            igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
            long neighborsSize = igraph_vector_size(&neighbors);
            if(params.fanOut < neighborsSize){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                status = mergeUDiagMatrix(dim[peerID], structure[peerID], structure[peerID]);
                if (status) {
                    exit(MergeError(__FUNCTION__));
                }
                memcpy(structure[neighborID][0], structure[peerID][0], dim[peerID] * dim[peerID] * sizeof(double));
                computeAverage(&dimestimate[peerID], dimestimate[neighborID]);
                dimestimate[neighborID] = dimestimate[peerID];
            }
            igraph_vector_destroy(&neighbors);
        }

        // check local convergence
        if (params.roundsToExecute < 0) {
            for(int peerID = 0; peerID < params.peers; peerID++){
                if(converged[peerID])
                    continue;
                bool dimestimateconv;
                if(prevestimate[peerID])
                    dimestimateconv = fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) < params.convThreshold;
                else
                    dimestimateconv = false;

                if(dimestimateconv)
                    convRounds[peerID]++;
                else
                    convRounds[peerID] = 0;

                converged[peerID] = (convRounds[peerID] >= params.convLimit);
                if(converged[peerID]){
                    Numberofconverged --;
                }
            }
        }
        rounds++;
        //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
        params.roundsToExecute--;
    }

    ON_EXIT:

    if (converged != nullptr)
        free(converged), converged = nullptr;

    if (convRounds != nullptr)
        free(convRounds), convRounds = nullptr;

    if (prevestimate != nullptr)
        free(prevestimate), prevestimate = nullptr;

    return dimestimate;
}

double* LocalSumAverageConsensus(Params params, igraph_t graph, int nCluster, double*** structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }
    double *dimestimate = nullptr, *prevestimate = nullptr;
    bool *converged = nullptr;
    int *convRounds = nullptr;
    int Numberofconverged = params.peers;
    int rounds = 0;
    int status = 0;

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(MemoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    while( (params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0){
        memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
        for(int peerID = 0; peerID < params.peers; peerID++){
            // check peer convergence
            if(params.roundsToExecute < 0 && converged[peerID])
                continue;
            // determine peer neighbors
            igraph_vector_t neighbors;
            igraph_vector_init(&neighbors, 0);
            igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
            long neighborsSize = igraph_vector_size(&neighbors);
            if(params.fanOut < neighborsSize){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                for (int l = 0; l < nCluster; ++l) {
                    status = computeAverage(&structure[peerID][0][l], structure[neighborID][0][l]);
                    if (status) {
                        exit(MergeError(__FUNCTION__));
                    }
                    status = computeAverage(&structure[peerID][1][l], structure[neighborID][1][l]);
                    if (status) {
                        exit(MergeError(__FUNCTION__));
                    }
                }
                memcpy(structure[neighborID][0], structure[peerID][0], 2 * nCluster * sizeof(double));
                computeAverage(&dimestimate[peerID], dimestimate[neighborID]);
                dimestimate[neighborID] = dimestimate[peerID];
            }
            igraph_vector_destroy(&neighbors);
        }

        // check local convergence
        if (params.roundsToExecute < 0) {
            for(int peerID = 0; peerID < params.peers; peerID++){
                if(converged[peerID])
                    continue;
                bool dimestimateconv;
                if(prevestimate[peerID])
                    dimestimateconv = fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) < params.convThreshold;
                else
                    dimestimateconv = false;

                if(dimestimateconv)
                    convRounds[peerID]++;
                else
                    convRounds[peerID] = 0;

                converged[peerID] = (convRounds[peerID] >= params.convLimit);
                if(converged[peerID]){
                    Numberofconverged --;
                }
            }
        }
        rounds++;
        //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
        params.roundsToExecute--;
    }

    ON_EXIT:

    if (converged != nullptr)
        free(converged), converged = nullptr;

    if (convRounds != nullptr)
        free(convRounds), convRounds = nullptr;

    if (prevestimate != nullptr)
        free(prevestimate), prevestimate = nullptr;

    return dimestimate;
}

double* CentroidsAverageConsensus(Params params, igraph_t graph, cube &structure) {
    if (structure.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }
    double *dimestimate = nullptr, *prevestimate = nullptr;
    bool *converged = nullptr;
    int *convRounds = nullptr;
    int Numberofconverged = params.peers;
    int rounds = 0;

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(MemoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    while( (params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0){
        memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
        for(int peerID = 0; peerID < params.peers; peerID++){
            // check peer convergence
            if(params.roundsToExecute < 0 && converged[peerID])
                continue;
            // determine peer neighbors
            igraph_vector_t neighbors;
            igraph_vector_init(&neighbors, 0);
            igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
            long neighborsSize = igraph_vector_size(&neighbors);
            if(params.fanOut < neighborsSize){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                structure.slice(peerID) = (structure.slice(peerID) + structure.slice(neighborID)) / 2;
                structure.slice(neighborID) = structure.slice(peerID);
                computeAverage(&dimestimate[peerID], dimestimate[neighborID]);
                dimestimate[neighborID] = dimestimate[peerID];
            }
            igraph_vector_destroy(&neighbors);
        }

        // check local convergence
        if (params.roundsToExecute < 0) {
            for(int peerID = 0; peerID < params.peers; peerID++){
                if(converged[peerID])
                    continue;
                bool dimestimateconv;
                if(prevestimate[peerID])
                    dimestimateconv = fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) < params.convThreshold;
                else
                    dimestimateconv = false;

                if(dimestimateconv)
                    convRounds[peerID]++;
                else
                    convRounds[peerID] = 0;

                converged[peerID] = (convRounds[peerID] >= params.convLimit);
                if(converged[peerID]){
                    Numberofconverged --;
                }
            }
        }
        rounds++;
        //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
        params.roundsToExecute--;
    }

    ON_EXIT:

    if (converged != nullptr)
        free(converged), converged = nullptr;

    if (convRounds != nullptr)
        free(convRounds), convRounds = nullptr;

    if (prevestimate != nullptr)
        free(prevestimate), prevestimate = nullptr;

    return dimestimate;
}