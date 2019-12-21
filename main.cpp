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
    string       inputFilename = "../datasets/HTRU_2.csv"; /**< The path for the input CSV file. */
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
    int          version = 1; /**< 0 for standard K-Means and 1 for K-Means++. */
};

chrono::high_resolution_clock::time_point t1, t2;

/**
 * This function stores the actual time into the global variable t1.
 */
void StartTheClock();
/**
 * This function stores the actual time into the global variable t2
 * and it computes the difference between t1 and t2.
 * @return the difference between t1 and t2
 */
double StopTheClock();
/**
 * This function handles the arguments passed on the command line
 * @param [in] argv it contains command line arguments.
 * @param [in] argc it is the count of command line arguments.
 * @param [in,out] params structure to store the arguments.
 * @return 0 if the command line arguments are ok, otherwise -5.
 */
int parseCommandLine(char **argv, int argc, Params &params);
/**
 * Print the needed parameters in order to run the application.
 * @param [in] cmd The name of the application.
 */
void usage(char* cmd);
/**
 * This function prints the parameters used for the run.
 * @param [in] params the structure holding the parameters.
 */
void printUsedParameters(Params params);
/**
 * This function computes the average between 2 values and the result is stored
 * in the first argument.
 * @param [in,out] x the address of the first value to be averaged.
 * @param [in] y the second value to be averaged.
 * @return 0 if it is correct, -2 if x is NULL.
 */
int computeAverage(double *x, double y);
/**
 * This function merges 2 arrays with point-wise average.
 * @param [in] dim the dimension of the array.
 * @param [in,out] peerVector the first array to be merged.
 * @param [in,out] neighborVector the second array to be merged.
 * @return 0 if it is correct, -2 if peerVector or neighborVector are NULL.
 */
int mergeVector(int dim, double *peerVector, double *neighborVector);
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
/**
 * This function computes the consensus on centroids owned by the peers according to the distance.
 * This function exits with code -2 if structure is empty or dist_vect is NULL and -1 for error
 * in memory allocation.
 * @param [in] params the structure containing the parameters for consensus.
 * @param [in] graph an igraph_vector_t structure (from igraph).
 * @param [in] structure a cube structure (from Armadillo library) [peers, 2, nCluster]
 *                          containing the information to be exchanged.
 * @param [in] dist_vect a vector containing the distance to compare the centroids to exchange.
 * @return an array [1, peers] containing the estimate for each peer of the total number
 *          of peers (to be used for estimation of the sum of the value in structure).
 */
double* MaxDistCentroidConsensus(Params params, igraph_t graph, cube &structure, double *dist_vect);
/**
 * This function merges the outliers count for all the peers. This function exits with
 * code -2 if structure is NULL and -1 for error in memory allocation.
 * @param [in] params the structure containing the parameters for consensus.
 * @param [in] graph an igraph_vector_t structure (from igraph).
 * @param [in] structure a structure of dimension [peers, peers] containing the
 *                          information to be exchanged.
 * @return an array [1, peers] containing the estimate for each peer of the total number
 *          of peers (to be used for estimation of the sum of the value in structure).
 */
double* OutliersConsensus(Params params, igraph_t graph, vector<vector<int>> *structure);
/**
 * This function writes the Outliers count of a single peer into the given CSV file.
 * @param [in] outliersCount a structure of dimension [peers, peers] containing the outliers information.
 * @param [in] n_subspaces the number of cnadidate subspaces.
 * @param [in] params the structure containing the parameters.
 * @return 0 if it is correct, -2 if outliersCount is empty.
 */
int writeOutliersOnCSV(vector<vector<int>> &outliersCount, int n_subspaces, Params params);

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
    double *dimestimate = NULL;
    bool *converged = NULL;
    int Numberofconverged;

    //Structures used for dataset loading and standardization
    double *data_storage = NULL, **data = NULL, *avg_storage = NULL, **avgsummaries = NULL;

    //Structures used for pcc and covariance computation
    double *pcc_storage = NULL, **pcc_i = NULL, ***pcc = NULL, **squaresum_dims = NULL,
            *squaresum_dims_storage = NULL, *covar_storage = NULL, **covar_i = NULL, ***covar = NULL;
    int *num_dims = NULL;

    //Structures used for Partitioning, PCA and Subspaces
    double ***combine = NULL, ***corr = NULL, ****subspace = NULL;
    int *uncorr_vars = NULL, *corr_vars = NULL, **uncorr = NULL;

    //Structures used for clustering
    double *centroids_dist = NULL, *localsum_storage = NULL, **localsum_i = NULL, ***localsum = NULL,
            *weights_storage = NULL, **weights = NULL, *prev_err = NULL, *error = NULL;
    cluster_report *final_i = NULL, **final = NULL, *prev = NULL;

    //Structures used for BetaCV
    double *pts_incluster_storage = NULL, **pts_incluster = NULL, *c_mean_storage = NULL, **c_mean = NULL,
            *BC_weight = NULL, *WC_weight = NULL, *Nin_edges = NULL, *Nout_edges = NULL;

    //Structures used for outlier identification
    double *inliers = NULL, *prev_inliers = NULL, *cluster_dim = NULL, *actual_dist = NULL,
            *actual_cluster_dim = NULL;
    bool ***discardedPts = NULL;
    vector< vector<int> > *outliers = NULL;

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
            programStatus = computeLocalAverage(data, n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID], avgsummaries[peerID]);
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
            programStatus = CenterData(avgsummaries[peerID], n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID], data);
        }
        if (programStatus) {
            goto ON_EXIT;
        }
    }

    free(avgsummaries), avgsummaries = NULL;
    free(avg_storage), avg_storage = NULL;

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
            programStatus = computeLocalPCC(pcc[peerID], squaresum_dims[peerID], n_dims, 0, peerLastItem[peerID], data);
        } else {
            programStatus = computeLocalPCC(pcc[peerID], squaresum_dims[peerID], n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID], data);
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
    free(num_dims), num_dims = NULL;

    for(int peerID = 0; peerID < params.peers; peerID++){
        programStatus = computePearsonMatrix(pcc[peerID], squaresum_dims[peerID], n_dims);
        if (programStatus) {
            goto ON_EXIT;
        }
    }

    free(squaresum_dims), squaresum_dims = NULL;
    free(squaresum_dims_storage), squaresum_dims_storage = NULL;

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to compute the Pearson matrix: " << elapsed << endl;
        cout << endl << "Partitioning dimensions in CORR and UNCORR sets..." << endl;
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

    uncorr = (int **) calloc(params.peers, sizeof(int *));
    if (!uncorr) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    corr = (double ***) calloc(params.peers, sizeof(double **));
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
            cout << "Correlated dimensions: " << corr_vars[peerID] << ", " << "Uncorrelated dimensions: " << uncorr_vars[peerID] << endl;

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
                    programStatus = copyDimension(data, corr_vars[peerID], dimensionID, 0, peerLastItem[peerID], corr[peerID]);
                } else {
                    programStatus = copyDimension(data, corr_vars[peerID], dimensionID, peerLastItem[peerID - 1] + 1, peerLastItem[peerID], corr[peerID]);
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

    free(pcc), pcc = NULL;
    free(pcc_i), pcc_i = NULL;
    free(pcc_storage), pcc_storage = NULL;

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
        programStatus = computeLocalCovarianceMatrix(partitionSize[peerID], corr_vars[peerID], corr[peerID], covar[peerID]);

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

        programStatus = computePCA(covar[peerID], corr[peerID], partitionSize[peerID], corr_vars[peerID], combine[peerID]);

        if (programStatus) {
            goto ON_EXIT;
        }

        for(int j = 0; j < corr_vars[peerID]; j++){
            free(corr[peerID][j]), corr[peerID][j] = NULL;
        }

        free(corr[peerID]), corr[peerID] = NULL;
    }

    free(corr), corr = NULL;
    free(covar), covar = NULL;
    free(covar_i), covar_i = NULL;
    free(covar_storage), covar_storage = NULL;
    free(corr_vars), corr_vars = NULL;

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
                    programStatus = copyDimension(data, 2, uncorr[peerID][subspaceID], 0, peerLastItem[peerID], combine[peerID]);
                } else {
                    programStatus = copyDimension(data, 2, uncorr[peerID][subspaceID], peerLastItem[peerID - 1] + 1, peerLastItem[peerID], combine[peerID]);
                }
                if (programStatus) {
                    goto ON_EXIT;
                }
            }

            programStatus = computeLocalCovarianceMatrix(partitionSize[peerID], 3, combine[peerID], covar[peerID]);

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
        free(num_dims), num_dims = NULL;

        for(int peerID = 0; peerID < params.peers; peerID++) {
            programStatus = computePCA(covar[peerID], combine[peerID], partitionSize[peerID], 3, subspace[peerID][subspaceID]);

            if (programStatus) {
                goto ON_EXIT;
            }
        }

        free(covar), covar = NULL;
        free(covar_i), covar_i = NULL;
        free(covar_storage), covar_storage = NULL;
    }


    for(int i = 0; i < params.peers; i++){
        free(uncorr[i]), uncorr[i] = NULL;
        for(int j = 0; j < 3; j++){
            free(combine[i][j]), combine[i][j] = NULL;
        }

        free(combine[i]), combine[i] = NULL;
    }

    free(uncorr), uncorr = NULL;
    if(combine != NULL)
        free(combine), combine = NULL;

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

        centroids_dist = (double *) calloc(params.peers, sizeof(double));
        if (!centroids_dist) {
            programStatus = MemoryError(__FUNCTION__);
            goto ON_EXIT;
        }

        for (int nCluster = 1; nCluster <= params.k_max; ++nCluster) {
            cube centroids(2, nCluster, params.peers, fill::zeros);
            if (params.version) {
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, n_data); // define the range
                int first_centroid = distr(eng);
                for(int peerID = 0; peerID < params.peers; peerID++) {
                    if (peerID == 0) {
                        if (first_centroid <= peerLastItem[peerID]) {
                            centroids(0, 0, peerID) = subspace[peerID][subspaceID][0][first_centroid];
                            centroids(1, 0, peerID) = subspace[peerID][subspaceID][1][first_centroid];
                            break;
                        }
                    }
                    else {
                        if (first_centroid >= peerLastItem[peerID - 1] + 1 && first_centroid <= peerLastItem[peerID]) {
                            int dataID = first_centroid - peerLastItem[peerID - 1] + 1;
                            centroids(0, 0, peerID) = subspace[peerID][subspaceID][0][dataID];
                            centroids(1, 0, peerID) = subspace[peerID][subspaceID][1][dataID];
                            break;
                        }
                    }
                }
                // Broadcast first centroid
                dimestimate = CentroidsAverageConsensus(params, graph, centroids);

                for(int peerID = 0; peerID < params.peers; peerID++){
                    centroids.slice(peerID) = centroids.slice(peerID) / dimestimate[peerID];
                }

                for (int centrToSet = 1; centrToSet < nCluster; ++centrToSet) {
                    for(int peerID = 0; peerID < params.peers; peerID++) {
                        programStatus = computeLocalInitialCentroid(centrToSet, partitionSize[peerID], subspace[peerID][subspaceID], centroids_dist[peerID], centroids.slice(peerID));

                        if (programStatus) {
                            goto ON_EXIT;
                        }
                    }
                    MaxDistCentroidConsensus(params, graph, centroids, centroids_dist);
                }
            }
            else {
                //Peer 0 sets random centroids
                centroids.slice(0) = randu<mat>(2, nCluster);
                // Broadcast random centroids
                dimestimate = CentroidsAverageConsensus(params, graph, centroids);

                for(int peerID = 0; peerID < params.peers; peerID++){
                    centroids.slice(peerID) = centroids.slice(peerID) / dimestimate[peerID];
                }
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
                                                       subspace[peerID][subspaceID], weights[peerID], localsum[peerID],
                                                       error[peerID]);

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

            free(localsum), localsum = NULL;
            free(localsum_i), localsum_i = NULL;
            free(localsum_storage), localsum_storage = NULL;
            free(weights), weights = NULL;
            free(weights_storage), weights_storage = NULL;
            free(prev_err), prev_err = NULL;
            free(error), error = NULL;

            for(int peerID = 0; peerID < params.peers; peerID++){
                final[subspaceID][peerID].centroids = centroids.slice(peerID);
                final[subspaceID][peerID].k = nCluster;
                if (nCluster == 1) {
                    final[subspaceID][peerID].BetaCV = 0.0;
                } else {
                    programStatus = create_cidx_matrix(subspace[peerID][subspaceID], partitionSize[peerID], final[subspaceID][peerID]);

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

                    programStatus = computeLocalC_Mean(subspace[peerID][subspaceID], partitionSize[peerID],  c_mean[peerID]);

                    if (programStatus) {
                        goto ON_EXIT;
                    }

                    programStatus = computeLocalPtsIncluster(partitionSize[peerID], pts_incluster[peerID], final[subspaceID][peerID]);

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
                    programStatus = computeNin_Nout(nCluster, pts_incluster[peerID], Nin_edges[peerID], Nout_edges[peerID]);

                    if (programStatus) {
                        goto ON_EXIT;
                    }

                    BC_weight[peerID] = computeBCSS(pts_incluster[peerID], final[subspaceID][peerID].centroids, c_mean[peerID]);
                    WC_weight[peerID] = computeLocalWCweight(partitionSize[peerID], final[subspaceID][peerID], subspace[peerID][subspaceID]);

                }

                free(pts_incluster), pts_incluster = NULL;
                free(pts_incluster_storage), pts_incluster_storage = NULL;
                free(c_mean), c_mean = NULL;
                free(c_mean_storage), c_mean_storage = NULL;

                dimestimate = SingleValueAverageConsensus(params, graph, WC_weight);

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    WC_weight[peerID] = WC_weight[peerID] / dimestimate[peerID];
                    final[subspaceID][peerID].BetaCV = (Nout_edges[peerID] * WC_weight[peerID]) / (Nin_edges[peerID] * BC_weight[peerID]);
                }

                free(BC_weight), BC_weight = NULL;
                free(WC_weight), WC_weight = NULL;
                free(Nin_edges), Nin_edges = NULL;
                free(Nout_edges), Nout_edges = NULL;

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

        free(prev), prev = NULL;
        free(centroids_dist), centroids_dist = NULL;
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

                    actual_cluster_dim[peerID] = computeLocalClusterDimension(partitionSize[peerID], final[subspaceID][peerID], clusterID, discardedPts[peerID][subspaceID]);

                    actual_dist[peerID] = computeLocalClusterDistance(partitionSize[peerID], final[subspaceID][peerID], clusterID, discardedPts[peerID][subspaceID], subspace[peerID][subspaceID]);

                }

                SingleValueAverageConsensus(params, graph, actual_dist);
                SingleValueAverageConsensus(params, graph, actual_cluster_dim);

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    double dist_mean = actual_dist[peerID] / actual_cluster_dim[peerID];
                    inliers[peerID] += computeLocalInliers(partitionSize[peerID], final[subspaceID][peerID], clusterID, discardedPts[peerID][subspaceID], subspace[peerID][subspaceID], dist_mean);

                }

                SingleValueAverageConsensus(params, graph, inliers);

                // check local convergence
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    if (converged[peerID])
                        continue;

                    converged[peerID] = ( (inliers[peerID] >= params.percentageIncircle * cluster_dim[peerID]) || prev_inliers[peerID] == inliers[peerID] );

                    if (converged[peerID]) {
                        Numberofconverged--;
                    }
                }
            }

            free(actual_dist), actual_dist = NULL;
            free(actual_cluster_dim), actual_cluster_dim = NULL;
        }

        free(inliers), inliers = NULL;
        free(prev_inliers), prev_inliers = NULL;
        free(cluster_dim), cluster_dim = NULL;
    }

    /***    Final result broadcast to all the peers
     * Each peer stores its outliers count in "outliers", then the global solution is merged
     * with a consensus on "outliers". Finally peer 0 is queried to get the final result.
    ***/
    outliers = new (nothrow) vector< vector<int> >[params.peers];
    if (!outliers) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    for (int peerID = 0; peerID < params.peers; peerID++) {
        outliers[peerID].resize(params.peers);

        programStatus = getCountOutliersinSubspace(uncorr_vars[peerID], partitionSize[peerID], discardedPts[peerID],
                                                   outliers[peerID][peerID]);

        if (programStatus) {
            goto ON_EXIT;
        }
    }

    dimestimate = OutliersConsensus(params, graph, outliers);

    if (!outputOnFile) {
        cout << endl << "OUTLIERS:" << endl;
        int index = 0;
        for (int peerID = 0; peerID < params.peers; peerID++) {
            for (int dataID = 0; dataID < outliers[0][peerID].size(); ++dataID) {
                if (outliers[0][peerID][dataID] >= std::round(uncorr_vars[0] * params.percentageSubspaces)) {
                    cout << index + dataID << " (" << outliers[0][peerID][dataID] << "), ";
                }
            }
            index += outliers[0][peerID].size();
        }
        cout << endl << "CENTROIDS:" << endl;
        for (int subspaceID = 0; subspaceID < uncorr_vars[0]; ++subspaceID) {
            final[subspaceID][0].centroids.print();
            cout << endl;
        }
    }
    writeOutliersOnCSV(outliers[0], uncorr_vars[0], params);

    elapsed = StopTheClock();
    if (!outputOnFile) {
        cout << "Time (seconds) required to identify the outliers: " << elapsed << endl;
    }

    igraph_destroy(&graph);
    programStatus = 0;

    ON_EXIT:

    if(data != NULL)
        free(data), data = NULL;
    if(data_storage != NULL)
        free(data_storage), data_storage = NULL;
    if(peerLastItem != NULL)
        free(peerLastItem), peerLastItem = NULL;
    if(partitionSize != NULL)
        free(partitionSize), partitionSize = NULL;
    if(converged != NULL)
        free(converged), converged = NULL;
    if(avgsummaries != NULL)
        free(avgsummaries), avgsummaries = NULL;
    if(avg_storage != NULL)
        free(avg_storage), avg_storage = NULL;
    if(pcc != NULL)
        free(pcc), pcc = NULL;
    if(pcc_i != NULL)
        free(pcc_i), pcc_i = NULL;
    if(pcc_storage != NULL)
        free(pcc_storage), pcc_storage = NULL;
    if(squaresum_dims != NULL)
        free(squaresum_dims), squaresum_dims = NULL;
    if(squaresum_dims_storage != NULL)
        free(squaresum_dims_storage), squaresum_dims_storage = NULL;
    if(num_dims != NULL)
        free(num_dims), num_dims = NULL;


    if(uncorr != NULL) {
        for(int i = 0; i < params.peers; i++){
            if(uncorr[i] != NULL)
                free(uncorr[i]), uncorr[i] = NULL;
        }

        free(uncorr), uncorr = NULL;

    }

    if(corr_vars != NULL) {
        if(corr != NULL)
            for(int i = 0; i < params.peers; i++){
                for(int j = 0; j < corr_vars[i]; j++){
                    if(corr[i] != NULL && corr[i][j] != NULL)
                        free(corr[i][j]), corr[i][j] = NULL;
                }
                if(corr[i] != NULL)
                    free(corr[i]), corr[i] = NULL;
            }

        free(corr), corr = NULL;
        free(corr_vars), corr_vars = NULL;
    }

    if(covar != NULL)
        free(covar), covar = NULL;
    if(covar_i != NULL)
        free(covar_i), covar_i = NULL;
    if(covar_storage != NULL)
        free(covar_storage), covar_storage = NULL;
    if(combine != NULL) {
        for(int i = 0; i < params.peers; i++){
            for(int j = 0; j < 3; j++){
                if(combine[i] != NULL && combine[i][j] != NULL)
                    free(combine[i][j]), combine[i][j] = NULL;
            }

            if(combine[i] != NULL)
                free(combine[i]), combine[i] = NULL;
        }

        free(combine), combine = NULL;
    }

    if(uncorr_vars != NULL) {
        if(subspace != NULL) {
            for(int i = 0; i < params.peers; i++){
                for(int j = 0; j < uncorr_vars[i]; j++){
                    for (int k = 0; k < 2; ++k) {
                        if(subspace[i] != NULL && subspace[i][j] != NULL && subspace[i][j][k] != NULL)
                            free(subspace[i][j][k]), subspace[i][j][k] = NULL;
                    }

                    if(subspace[i] != NULL && subspace[i][j] != NULL)
                        free(subspace[i][j]), subspace[i][j] = NULL;
                }

                if(subspace[i] != NULL)
                    free(subspace[i]), subspace[i] = NULL;
            }
        }

        free(subspace), subspace = NULL;
        if(discardedPts != NULL) {
            for(int i = 0; i < params.peers; i++){
                for(int j = 0; j < uncorr_vars[i]; j++){
                    if(discardedPts[i] != NULL && discardedPts[i][j] != NULL)
                        free(discardedPts[i][j]), discardedPts[i][j] = NULL;
                }

                if(discardedPts[i] != NULL)
                    free(discardedPts[i]), discardedPts[i] = NULL;
            }

            free(discardedPts), discardedPts = NULL;
        }

        free(uncorr_vars), uncorr_vars = NULL;
    }

    if(final_i != NULL)
        free(final_i), final_i = NULL;
    if(final != NULL)
        free(final), final = NULL;
    if(prev != NULL)
        free(prev), prev = NULL;
    if(centroids_dist != NULL)
        free(centroids_dist), centroids_dist = NULL;
    if(localsum != NULL)
        free(localsum), localsum = NULL;
    if(localsum_i != NULL)
        free(localsum_i), localsum_i = NULL;
    if(localsum_storage != NULL)
        free(localsum_storage), localsum_storage = NULL;
    if(weights != NULL)
        free(weights), weights = NULL;
    if(weights_storage != NULL)
        free(weights_storage), weights_storage = NULL;
    if(prev_err != NULL)
        free(prev_err), prev_err = NULL;
    if(error != NULL)
        free(error), error = NULL;
    if(pts_incluster != NULL)
        free(pts_incluster), pts_incluster = NULL;
    if(pts_incluster_storage != NULL)
        free(pts_incluster_storage), pts_incluster_storage = NULL;
    if(c_mean != NULL)
        free(c_mean), c_mean = NULL;
    if(c_mean_storage != NULL)
        free(c_mean_storage), c_mean_storage = NULL;
    if(BC_weight != NULL)
        free(BC_weight), BC_weight = NULL;
    if(WC_weight != NULL)
        free(WC_weight), WC_weight = NULL;
    if(Nin_edges != NULL)
        free(Nin_edges), Nin_edges = NULL;
    if(Nout_edges != NULL)
        free(Nout_edges), Nout_edges = NULL;
    if(inliers != NULL)
        free(inliers), inliers = NULL;
    if(prev_inliers != NULL)
        free(prev_inliers), prev_inliers = NULL;
    if(cluster_dim != NULL)
        free(cluster_dim), cluster_dim = NULL;
    if(actual_dist != NULL)
        free(actual_dist), actual_dist = NULL;
    if(actual_cluster_dim != NULL)
        free(actual_cluster_dim), actual_cluster_dim = NULL;
    if(outliers != NULL) {
        delete[] outliers, outliers = NULL;
    }
    if(dimestimate != NULL)
        free(dimestimate), dimestimate = NULL;

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
                return ArgumentError(__FUNCTION__);
            }
            params.peers = stoi(argv[i]);
        } else if (strcmp(argv[i], "-f") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing fan-out parameter." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.fanOut = stol(argv[i]);
        } else if (strcmp(argv[i], "-d") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing graph type parameter." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.graphType = stoi(argv[i]);
        } else if (strcmp(argv[i], "-ct") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing convergence tolerance parameter." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.convThreshold = stod(argv[i]);
        } else if (strcmp(argv[i], "-cl") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing # of consecutive rounds in which convergence is satisfied parameter." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.convLimit = stol(argv[i]);
        } else if (strcmp(argv[i], "-of") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing filename for simulation output." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.outputFilename = string(argv[i]);
        } else if (strcmp(argv[i], "-r") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of rounds to execute." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.roundsToExecute = stoi(argv[i]);
        } else if (strcmp(argv[i], "-k") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing max number of clusters for Elbow method." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.k_max = stol(argv[i]);
        } else if (strcmp(argv[i], "-et") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing threshold for Elbow method." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.convClusteringThreshold = stof(argv[i]);
        } else if (strcmp(argv[i], "-clst") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing threshold for distributed clustering." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.elbowThreshold = stof(argv[i]);
        } else if (strcmp(argv[i], "-pi") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of inlier points." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.percentageIncircle = stof(argv[i]);
        } else if (strcmp(argv[i], "-ps") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of subspace in which an outlier must be." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.percentageSubspaces = stof(argv[i]);
        } else if (strcmp(argv[i], "-if") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing input file name." << endl;
                return ArgumentError(__FUNCTION__);
            }
            params.inputFilename = string(argv[i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing K-Means version.\n";
                return ArgumentError(__FUNCTION__);
            }
            params.inputFilename = string(argv[i]);
        } else {
            usage(argv[0]);
            return ArgumentError(__FUNCTION__);
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
            << "-d          graph type: 1 geometric 2 Barabasi-Albert 3 Erdos-Renyi 4 regular" << endl
            << "-ct         convergence tolerance" << endl
            << "-cl         number of consecutive rounds in which convergence must be satisfied" << endl
            << "-of         output filename, if specified a file with this name containing all of the peers stats is written" << endl
            << "-r          max number of rounds to execute in consensus algorithm" << endl
            << "-k          max number of clusters to try in elbow criterion" << endl
            << "-et         threshold for the selection of optimal number of clusters in Elbow method" << endl
            << "-clst       the local convergence tolerance for distributed K-Means" << endl
            << "-pi         percentage of points in a cluster to be evaluated as inlier" << endl
            << "-ps         percentage of subspaces in which a point must be outlier to be evaluated as general outlier" << endl
            << "-if         input filename" << endl
            << "-v          K-Means version: 0 for standard K-Means and 1 for K-Means++" << endl << endl;
}

void printUsedParameters(Params params) {
    cout << endl << endl << "PARAMETERS: " << endl
         << "input file= " << params.inputFilename  << endl
         << "percentage in circle = " << params.percentageIncircle  << endl
         << "elbow threshold = " << params.elbowThreshold  << endl
         << "convergence clustering threshold = " << params.convClusteringThreshold  << endl
         << "percentage subspaces = " << params.percentageSubspaces  << endl
         << "k_max = " << params.k_max  << endl
         << "K-Means version = " << params.version << endl
         << "local convergence tolerance = "<< params.convThreshold  << endl
         << "number of consecutive rounds in which a peer must locally converge = "<< params.convLimit << endl
         << "peers = " << params.peers  << endl
         << "fan-out = " << params.fanOut << endl
         << "graph type = ";
    printGraphType(params.graphType);
    cout << endl << endl;

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

double* SingleValueAverageConsensus(Params params, igraph_t graph, double* structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }

    double *dimestimate = NULL, *prevestimate = NULL;
    bool *converged = NULL;
    int *convRounds = NULL;
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
            if(params.fanOut < neighborsSize && params.fanOut != -1){
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

    if (converged != NULL)
        free(converged), converged = NULL;

    if (convRounds != NULL)
        free(convRounds), convRounds = NULL;

    if (prevestimate != NULL)
        free(prevestimate), prevestimate = NULL;

    return dimestimate;
}

double* VectorAverageConsensus(Params params, igraph_t graph, int dim, double** structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }

    double *dimestimate = NULL, *prevestimate = NULL;
    bool *converged = NULL;
    int *convRounds = NULL;
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
            if(params.fanOut < neighborsSize && params.fanOut != -1){
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

    if (converged != NULL)
        free(converged), converged = NULL;

    if (convRounds != NULL)
        free(convRounds), convRounds = NULL;

    if (prevestimate != NULL)
        free(prevestimate), prevestimate = NULL;

    return dimestimate;
}

double* UDiagMatrixAverageConsensus(Params params, igraph_t graph, int *dim, double*** structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }

    double *dimestimate = NULL, *prevestimate = NULL;
    bool *converged = NULL;
    int *convRounds = NULL;
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
            if(params.fanOut < neighborsSize && params.fanOut != -1){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                for (int l = 0; l < dim[peerID]; ++l) {
                    for (int k = l; k < dim[peerID]; ++k) {
                        status = computeAverage(&structure[peerID][l][k], structure[neighborID][l][k]);
                        if (status) {
                            exit(MergeError(__FUNCTION__));
                        }

                        structure[peerID][k][l] = structure[peerID][l][k];
                    }
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

    if (converged != NULL)
        free(converged), converged = NULL;

    if (convRounds != NULL)
        free(convRounds), convRounds = NULL;

    if (prevestimate != NULL)
        free(prevestimate), prevestimate = NULL;

    return dimestimate;
}

double* LocalSumAverageConsensus(Params params, igraph_t graph, int nCluster, double*** structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }

    double *dimestimate = NULL, *prevestimate = NULL;
    bool *converged = NULL;
    int *convRounds = NULL;
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
            if(params.fanOut < neighborsSize && params.fanOut != -1){
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

    if (converged != NULL)
        free(converged), converged = NULL;

    if (convRounds != NULL)
        free(convRounds), convRounds = NULL;

    if (prevestimate != NULL)
        free(prevestimate), prevestimate = NULL;

    return dimestimate;
}

double* CentroidsAverageConsensus(Params params, igraph_t graph, cube &structure) {
    if (structure.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    double *dimestimate = NULL, *prevestimate = NULL;
    bool *converged = NULL;
    int *convRounds = NULL;
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
            if(params.fanOut < neighborsSize && params.fanOut != -1){
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

    if (converged != NULL)
        free(converged), converged = NULL;

    if (convRounds != NULL)
        free(convRounds), convRounds = NULL;

    if (prevestimate != NULL)
        free(prevestimate), prevestimate = NULL;

    return dimestimate;
}

double* MaxDistCentroidConsensus(Params params, igraph_t graph, cube &structure, double *dist_vect) {
    if (structure.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    double *dimestimate = NULL, *prevestimate = NULL;
    bool *converged = NULL;
    int *convRounds = NULL;
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
            if(params.fanOut < neighborsSize && params.fanOut != -1){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);
                if (dist_vect[peerID] > dist_vect[neighborID]) {
                    structure.slice(peerID) = structure.slice(neighborID);
                    dist_vect[peerID] = dist_vect[neighborID];
                }
                else {
                    structure.slice(neighborID) = structure.slice(peerID);
                    dist_vect[neighborID] = dist_vect[peerID];
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

    if (converged != NULL)
        free(converged), converged = NULL;

    if (convRounds != NULL)
        free(convRounds), convRounds = NULL;

    if (prevestimate != NULL)
        free(prevestimate), prevestimate = NULL;

    return dimestimate;
}

double* OutliersConsensus(Params params, igraph_t graph, vector<vector<int>> *structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }

    double *dimestimate = NULL, *prevestimate = NULL;
    bool *converged = NULL;
    int *convRounds = NULL;
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
            if(params.fanOut < neighborsSize && params.fanOut != -1){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                for (int k = 0; k < params.peers; ++k) {
                    if (structure[peerID][k].empty()) {
                        structure[peerID][k] = structure[neighborID][k];
                    }
                    structure[neighborID][k] = structure[peerID][k];
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

    if (converged != NULL)
        free(converged), converged = NULL;

    if (convRounds != NULL)
        free(convRounds), convRounds = NULL;

    if (prevestimate != NULL)
        free(prevestimate), prevestimate = NULL;

    return dimestimate;
}

int writeOutliersOnCSV(vector<vector<int>> &outliersCount, int n_subspaces, Params params) {
    if (outliersCount.empty()) {
        return NullPointerError(__FUNCTION__);
    }

    fstream fout;
    fout.open(params.outputFilename, ios::out | ios::trunc);

    int index = 0;
    for (int peerID = 0; peerID < params.peers; peerID++) {
        for (int dataID = 0; dataID < outliersCount[peerID].size(); ++dataID) {
            if (outliersCount[peerID][dataID] >= std::round(n_subspaces * params.percentageSubspaces)) {
                fout << index + dataID << ",";
            }
        }
        index += outliersCount[peerID].size();
    }
    fout.close();
    return 0;
}