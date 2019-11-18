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
    int          peers; /**< The number of peers. */
    string       inputFilename; /**< The path for the input CSV file. */
    string       outputFilename; /**< The path for the output file. */
    double       convThreshold; /**< The local convergence tolerance for the consensus algorithm. */
    int          convLimit; /**< The number of consecutive rounds in which a peer must locally converge. */
    int          graphType; /**< The graph distribution: 1 Geometric, 2 Barabasi-Albert, 3 Erdos-Renyi, 4 Regular (clique) */
    int          fanOut; /**< The number of communication that a peer can carry out in a round. */
    int          roundsToExecute; /**< The number of rounds to carry out in the consensus algorithm. */
    long         k_max; /**< The maximum number of cluster to try for the K-Means algorithm. */
    double       elbowThreshold; /**< The error tolerance for the selected metric to evaluate the elbow in K-means algorithm. */
    double       convClusteringThreshold; /**< The local convergence tolerance for distributed K-Means. */
    double       percentageIncircle; /**< The percentage of points in a cluster to be considered as inliers. */
    double       percentageSubspaces; /**< The percentage of subspace in which a point must be outlier to be evaluated as general outlier. */
};

/**
 * Print the needed parameters in order to run the script
 * @param cmd The name of the script.
 */
void usage(char* cmd);
int partitionData(int n_data, int peers, long *peerLastItem, long *partitionSize);
void printUsedParameters(Params params);
int computeLocalAverage(double **data, int ndims, long start, long end, double *summaries);
int CenterData(double *summaries, int ndims, long start, long end, double **data);
int computeLocalPCC(double **pcc, double *squaresum_dims, int ndims, long start, long end, double **data);
int computePearsonMatrix(double **pcc, double *squaresum_dims, int ndims);
bool isCorrDimension(int ndims, int firstDimension, double **pcc);
int computeCorrUncorrCardinality(double **pcc, int ndims, int &corr_vars, int &uncorr_vars);
int copyDimension(double **data, int corr_vars, int dim, long start, long end, double **newstorage);
int computeLocalCovarianceMatrix(long partitionSize, int corr_vars, double **corrSet, double **covarianceMatrix);
int computePCA(double **covarianceMatrix, double **oldSpace, long partitionSize, int n_dims, double **newSpace);
int computeLocalKMeans(long partitionSize, mat centroids, double **subspace, double *weights, double **localsum, double &error);
int computeLocalC_Mean(double **data, long partitionSize, double *summaries);
int computeLocalPtsIncluster(long partitionSize, double *pts_incluster, cluster_report rep);
int computeNin_Nout(int nCluster, double *pts_incluster, double &Nin, double &Nout);
double computeBCSS(double *pts_incluster, mat centroids, double *c_mean);
double computeLocalWCSS(long partitionSize, cluster_report rep, double **subspace);
double computeLocalClusterDistance(long partitionSize, cluster_report rep, int clusterid, bool *discarded, double **subspace);
int computeLocalClusterDimension(long partitionSize, cluster_report rep, int clusterid, bool *discarded);
int computeLocalInliers(long partitionSize, cluster_report rep, int clusterid, bool *discarded, double **subspace, double radius);
int getCountOutliersinSubspace(int uncorr_vars, long partitionSize, int start_idx, bool **discarded, double *outliers);
int computeAverage(double *x, double y);
int mergeSimpleMessages(int dim, double *peerElem, double *neighborElem);
int mergeUDiagMatrix(int n_dims, double **UDiagMatrix_X, double **UDiagMatrix_Y);
double* SimpleAverageConsensus(Params params, igraph_t graph, double* structure);
double* VectorAverageConsensus(Params params, igraph_t graph, int dim, double** structure);
double* UDiagMatrixAverageConsensus(Params params, igraph_t graph, int *dim, double*** structure);
double* LocalSumAverageConsensus(Params params, igraph_t graph, int nCluster, double*** structure);
double* CentroidsAverageConsensus(Params params, igraph_t graph, cube &structure);

int main(int argc, char **argv) {

    int n_dims; // number of dimensions
    int n_data; // number of data
    long *peerLastItem = NULL; // index of a peer last item
    long *partitionSize = NULL; // size of a peer partition
    int peers = 10; // number of peers
    int fanOut = 3; //fan-out of peers
    uint32_t seed = 16033099; // seed for the PRNG
    int graphType = 2; // graph distribution: 1 geometric 2 Barabasi-Albert 3 Erdos-Renyi 4 regular (clique)
    double convThreshold = 0.001; // local convergence tolerance
    int convLimit = 3; // number of consecutive rounds in which a peer must locally converge
    int roundsToExecute = -1;
    long k_max = 10; // max number of clusters to try in elbow criterion
    double elbowThreshold = 0.25; // threshold for the selection of optimal number of clusters in Elbow method
    double convClusteringThreshold = 0.0001; // local convergence tolerance for distributed clustering
    double percentageIncircle = 0.9; // percentage of points in a cluster to be evaluated as inlier
    double percentageSubspaces = 0.8; // percentage of subspaces in which a point must be outlier to be evaluated as general outlier

    Params          params;
    bool            outputOnFile = false;
    string          inputFilename = "../datasets/Iris.csv";
    string          outputFilename;
    igraph_t        graph;

    /*** Parse Command-Line Parameters ***/
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-p") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of peers parameter." << endl;
                return -1;
            }
            peers = stoi(argv[i]);
        } else if (strcmp(argv[i], "-f") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing fan-out parameter." << endl;
                return -1;
            }
            fanOut = stol(argv[i]);;
        } else if (strcmp(argv[i], "-d") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing graph type parameter" << endl;
                return -1;
            }
            graphType = stoi(argv[i]);
        } else if (strcmp(argv[i], "-ct") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing convergence tolerance parameter." << endl;
                return -1;
            }
            convThreshold = stod(argv[i]);
        } else if (strcmp(argv[i], "-cl") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing # of consecutive rounds in which convergence is satisfied parameter." << endl;
                return -1;
            }
            convLimit = stol(argv[i]);
        } else if (strcmp(argv[i], "-of") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing filename for simulation output." << endl;
                return -1;
            }
            outputFilename = string(argv[i]);
        } else if (strcmp(argv[i], "-r") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of rounds to execute.\n";
                return -1;
            }
            roundsToExecute = stoi(argv[i]);
        } else if (strcmp(argv[i], "-k") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing max number of clusters for Elbow method.\n";
                return -1;
            }
            k_max = stol(argv[i]);
        } else if (strcmp(argv[i], "-et") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing threshold for Elbow method.\n";
                return -1;
            }
            convClusteringThreshold = stof(argv[i]);
        } else if (strcmp(argv[i], "-clst") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing threshold for distributed clustering.\n";
                return -1;
            }
            elbowThreshold = stof(argv[i]);
        } else if (strcmp(argv[i], "-pi") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of inlier points.\n";
                return -1;
            }
            percentageIncircle = stof(argv[i]);
        } else if (strcmp(argv[i], "-ps") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of subspace in which an outlier must be.\n";
                return -1;
            }
            percentageSubspaces = stof(argv[i]);
        } else if (strcmp(argv[i], "-if") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing input file name.\n";
                return -1;
            }
            inputFilename = string(argv[i]);
        } else {
            usage(argv[0]);
            return -1;
        }
    }

    /*** Dataset Loading ***/
    getDatasetDims(inputFilename, n_dims, n_data);

    double *data_storage = (double *) malloc(n_dims * n_data * sizeof(double));
    if (!data_storage) {
        cerr << "Malloc error on data_storage" << endl;
        exit(-1);
    }
    double **data = (double **) malloc(n_data * sizeof(double *));
    if (!data) {
        cerr << "Malloc error on data" << endl;
        exit(-1);
    }
    for (int i = 0; i < n_data; ++i) {
        data[i] = &data_storage[i * n_dims];
    }

    if (loadData(inputFilename, data, n_dims)) {
        cerr << "Error on loading dataset" << endl;
        exit(-1);
    }

    /*** Partitioning phase ***/
    peerLastItem = (long *) calloc(peers, sizeof(long));
    if (!peerLastItem) {
        cerr << "Malloc error on peerLastItem" << endl;
        exit(-1);
    }
    partitionSize = (long *) calloc(peers, sizeof(long));
    if (!partitionSize) {
        cerr << "Malloc error on partitionSize" << endl;
        exit(-1);
    }
    partitionData(n_data, peers, peerLastItem, partitionSize);

    /*** Assign parameters read from command line ***/
    params.peers = peers;
    params.fanOut = fanOut;
    params.graphType = graphType;
    params.convThreshold = convThreshold;
    params.convLimit = convLimit;
    params.outputFilename = outputFilename;
    params.roundsToExecute = roundsToExecute;
    params.elbowThreshold = elbowThreshold;
    params.convClusteringThreshold = convClusteringThreshold;
    params.k_max = k_max;
    params.percentageIncircle = percentageIncircle;
    params.percentageSubspaces = percentageSubspaces;
    params.inputFilename = inputFilename;
    outputOnFile = params.outputFilename.size() > 0;

    if (!outputOnFile) {
        printUsedParameters(params);
    }

    /*** Graph generation ***/
    auto gengraphstart = chrono::steady_clock::now();

    graph = generateGraph(params.graphType, params.peers);

    auto gengraphend = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to generate the random graph: " <<
             chrono::duration_cast<chrono::nanoseconds>(gengraphend - gengraphstart).count()*1e-9 << endl;
    }

    // determine minimum and maximum vertex degree for the graph
    igraph_vector_t result;
    result = getMinMaxVertexDeg(graph, outputOnFile);
    igraph_vector_destroy(&result);

    // this is used to estimate the number of peers
    double *dimestimate;

    int Numberofconverged = params.peers;
    bool *converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        cerr << "Malloc error on converged" << endl;
        exit(-1);
    }

    if (!outputOnFile) {
        printf("\nApplying Dataset Standardization to each peer' substream...\n");
    }

    /***    Dataset Standardization
     * The dataset is centered around the mean value.
     * 1) Each peer computes the local average, for each dimension, on its dataset
     * partition and saves the values on "avgsummaries".
     * 2) An average consensus is executed on the average value for each dimension.
     * 3) Each peer centers its dataset partition on the average reached with consensus.
    ***/
    auto std_start = chrono::steady_clock::now();
    double *avg_storage = (double *) calloc(params.peers * n_dims, sizeof(double));
    if (!avg_storage) {
        cerr << "Malloc error on avg_storage" << endl;
        exit(-1);
    }
    double **avgsummaries = (double **) malloc(params.peers * sizeof(double *));
    if (!avgsummaries) {
        cerr << "Malloc error on avgsummaries" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        avgsummaries[peerID] = &avg_storage[peerID * n_dims];

        if (peerID == 0) {
            computeLocalAverage(data, n_dims, 0, peerLastItem[peerID], avgsummaries[peerID]);
        } else {
            computeLocalAverage(data, n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID], avgsummaries[peerID]);
        }
    }

    VectorAverageConsensus(params, graph, n_dims, avgsummaries);

    for(int peerID = 0; peerID < params.peers; peerID++){
        if (peerID == 0) {
            CenterData(avgsummaries[peerID], n_dims, 0, peerLastItem[peerID], data);
        } else {
            CenterData(avgsummaries[peerID], n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID], data);
        }
    }
    free(avgsummaries);
    free(avg_storage);

    auto std_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to standardize the dataset: " <<
             chrono::duration_cast<chrono::nanoseconds>(std_end - std_start).count()*1e-9 << endl;
    }

    if (!outputOnFile) {
        printf("\nComputing Pearson matrix globally...\n");
    }

    /***    Pearson Matrix Computation
     * The Pearson Coefficient between two dimensions x and y when
     * the dataset dimensions are centered around the mean values is:
     * r_xy = sum_i of (x_i * y_i) / sqrt(sum_i of pow2(x_i) * sum_i of pow2(y_i))
     * 1) Locally each peer computes the numerator (on pcc structure) and
     *      the denominator (on squaresum_dims) of the previous r_xy for
     *      each pair of dimensions.
     * 2) A consensus on the sum of pcc and squaresum_dims is executed.
     * 3) Each peer computes the Pearson matrix with the values resulting
     *      from consensus.
    ***/
    auto pcc_start = chrono::steady_clock::now();
    double *pcc_storage, **pcc_i, ***pcc, **squaresum_dims, *squaresum_dims_storage;
    pcc_storage = (double *) calloc(params.peers * n_dims * n_dims, sizeof(double));
    if (!pcc_storage) {
        cerr << "Malloc error on pcc_storage" << endl;
        exit(-1);
    }
    pcc_i = (double **) malloc(params.peers * n_dims * sizeof(double *));
    if (!pcc_i) {
        cerr << "Malloc error on pcc_i" << endl;
        exit(-1);
    }
    pcc = (double ***) malloc(params.peers * sizeof(double **));
    if (!pcc) {
        cerr << "Malloc error on pcc" << endl;
        exit(-1);
    }
    squaresum_dims_storage = (double *) calloc(params.peers * n_dims, sizeof(double));
    if (!squaresum_dims_storage) {
        cerr << "Malloc error on squaresum_dims_storage" << endl;
        exit(-1);
    }
    squaresum_dims = (double **) malloc(params.peers * sizeof(double *));
    if (!squaresum_dims) {
        cerr << "Malloc error on squaresum_dims" << endl;
        exit(-1);
    }
    for (int i = 0; i < params.peers * n_dims; ++i) {
        pcc_i[i] = &pcc_storage[i * n_dims];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        pcc[peerID] = &pcc_i[peerID * n_dims];
        squaresum_dims[peerID] = &squaresum_dims_storage[peerID * n_dims];

        if (peerID == 0) {
            computeLocalPCC(pcc[peerID], squaresum_dims[peerID], n_dims, 0, peerLastItem[peerID], data);
        } else {
            computeLocalPCC(pcc[peerID], squaresum_dims[peerID], n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID], data);
        }
    }

    VectorAverageConsensus(params, graph, n_dims, squaresum_dims);
    int *num_dims = (int *) malloc(params.peers * sizeof(int));
    if (!num_dims) {
        cerr << "Malloc error on num_dims" << endl;
        exit(-1);
    }
    fill_n(num_dims, params.peers, n_dims);
    UDiagMatrixAverageConsensus(params, graph, num_dims, pcc);
    free(num_dims);

    for(int peerID = 0; peerID < params.peers; peerID++){
        computePearsonMatrix(pcc[peerID], squaresum_dims[peerID], n_dims);
    }
    free(squaresum_dims);
    free(squaresum_dims_storage);

    auto pcc_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to compute Pearson matrix: " <<
             chrono::duration_cast<chrono::nanoseconds>(pcc_end - pcc_start).count()*1e-9 << endl;
    }

    if (!outputOnFile) {
        printf("\nPartitioning dimensions in CORR and UNCORR sets...\n");
    }

    /***    CORR And UNCORR Partitioning
     * Locally each peer partition the dimensions in CORR and UNCORR sets
     * based on the row-wise sum of the Pearson coefficient for each dimension.
     * The structures corr_vars and uncorr_vars keep records of the
     * cardinality of CORR and UNCORR sets, while "corr" contains the peers'
     * dataset partition of a dimension and "uncorr" contains the indexes of
     * dimensions with Pearson coefficient less than 0.
    ***/
    auto partition_start = chrono::steady_clock::now();
    int *uncorr_vars, *corr_vars;
    corr_vars = (int *) calloc(params.peers, sizeof(int));
    if (!corr_vars) {
        cerr << "Malloc error on corr_vars" << endl;
        exit(-1);
    }
    uncorr_vars = (int *) calloc(params.peers, sizeof(int));
    if (!uncorr_vars) {
        cerr << "Malloc error on uncorr_vars" << endl;
        exit(-1);
    }
    int **uncorr = (int **) malloc(params.peers * sizeof(int *));
    if (!uncorr) {
        cerr << "Malloc error on corr or uncorr" << endl;
        exit(-1);
    };
    double ***corr = (double ***) malloc(params.peers * sizeof(double **));
    if (!corr) {
        cerr << "Malloc error on corr" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        computeCorrUncorrCardinality(pcc[peerID], n_dims, corr_vars[peerID], uncorr_vars[peerID]);

        if (peerID == 0) {
            cout << "Correlated dimensions: " << corr_vars[peerID] << ", " << "Uncorrelated dimensions: " << uncorr_vars[peerID] << endl;
            if (corr_vars[peerID] < 2) {
                cerr << "Correlated dimensions must be more than 1 in order to apply PCA!" << endl;
                exit(-1);
            }
            if (uncorr_vars[peerID] == 0) {
                cerr << "There are no candidate subspaces!" << endl;
                exit(-1);
            }
        }

        corr[peerID] = (double **) malloc(corr_vars[peerID] * sizeof(double *));
        if (!corr[peerID]) {
            cerr << "Malloc error on corr for peer " << peerID << endl;
            exit(-1);
        }
        for (int k = 0; k < corr_vars[peerID]; ++k) {
            corr[peerID][k] = (double *) malloc(partitionSize[peerID] * sizeof(double));
            if (!corr[peerID][k]) {
                cerr << "Malloc error on corr for peer " << peerID << endl;
                exit(-1);
            }
        }

        uncorr[peerID] = (int *) malloc(uncorr_vars[peerID] * sizeof(int));
        if (!uncorr[peerID]) {
            cerr << "Malloc error on uncorr" << endl;
            exit(-1);
        };

        corr_vars[peerID] = 0, uncorr_vars[peerID] = 0;
        for (int i = 0; i < n_dims; ++i) {
            if (isCorrDimension(n_dims, i, pcc[peerID])) {
                if (peerID == 0) {
                    copyDimension(data, corr_vars[peerID], i, 0, peerLastItem[peerID], corr[peerID]);
                } else {
                    copyDimension(data, corr_vars[peerID], i, peerLastItem[peerID-1] + 1, peerLastItem[peerID], corr[peerID]);
                }
                corr_vars[peerID]++;
            } else {
                uncorr[peerID][uncorr_vars[peerID]] = i;
                uncorr_vars[peerID]++;
            }
        }
    }
    free(pcc);
    free(pcc_i);
    free(pcc_storage);

    auto partition_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to partition the dataset in CORR and UNCORR sets: " <<
             chrono::duration_cast<chrono::nanoseconds>(partition_end - partition_start).count() * 1e-9 << endl;
    }

    if (!outputOnFile) {
        printf("\nComputing Principal Component Analysis on CORR set...\n");
    }

    /***    Principal Component Analysis on CORR set
     * 1) Locally each peer computes the covariance matrix on its dataset
     *      partition of CORR set and saves the result in "covar".
     * 2) An average consensus on the covariance matrix is executed.
     * 3) Locally each peer computes eigenvalues/eigenvectors (with Armadillo
     *      functions), get the 2 Principal Components and save them
     *      in "combine".
    ***/
    auto pca_start = chrono::steady_clock::now();
    double *covar_storage, **covar_i, ***covar;
    covar_storage = (double *) malloc(params.peers * corr_vars[0] * corr_vars[0] * sizeof(double));
    if (!covar_storage) {
        cerr << "Malloc error on covar_storage" << endl;
        exit(-1);
    }
    covar_i = (double **) malloc(params.peers * corr_vars[0] * sizeof(double *));
    if (!covar_i) {
        cerr << "Malloc error on covar_i" << endl;
        exit(-1);
    }
    covar = (double ***) malloc(params.peers * sizeof(double **));
    if (!covar) {
        cerr << "Malloc error on covar" << endl;
        exit(-1);
    }
    for (int i = 0; i < params.peers * corr_vars[0]; ++i) {
        covar_i[i] = &covar_storage[i * corr_vars[0]];
    }
    for (int i = 0; i < params.peers; ++i) {
        covar[i] = &covar_i[i * corr_vars[0]];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        computeLocalCovarianceMatrix(partitionSize[peerID], corr_vars[peerID], corr[peerID], covar[peerID]);
    }

    UDiagMatrixAverageConsensus(params, graph, corr_vars, covar);

    double ***combine;
    combine = (double ***) malloc(params.peers * sizeof(double **));
    if (!combine) {
        cerr << "Malloc error on combine" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        combine[peerID] = (double **) malloc(3 * sizeof(double *));
        if (!combine[peerID]) {
            cerr << "Malloc error on combine for peer " << peerID << endl;
            exit(-1);
        }
        for (int k = 0; k < 3; ++k) {
            combine[peerID][k] = (double *) malloc(partitionSize[peerID] * sizeof(double));
            if (!combine[peerID][k]) {
                cerr << "Malloc error on combine for peer " << peerID << endl;
                exit(-1);
            }
        }

        computePCA(covar[peerID], corr[peerID], partitionSize[peerID], corr_vars[peerID], combine[peerID]);
    }
    free(corr);
    free(corr_vars);
    free(covar_storage);
    free(covar_i);
    free(covar);

    auto pca_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to compute Principal Component Analysis on CORR set: " <<
             chrono::duration_cast<chrono::nanoseconds>(pca_end - pca_start).count()*1e-9 << endl;
    }

    if (!outputOnFile) {
        printf("\nComputing Candidate Subspaces through Principal Component Analysis"
               " on PC1corr, PC2corr and the m-th dimension in UNCORR set...\n");
    }

    /***    Candidate Subspaces Creation
     * 1) Locally each peer copies its dataset partition of the m-th
     *      dimension in UNCORR set and computes the covariance matrix.
     * 2) An average consensus on the covariance matrix is executed.
     * 3) Locally each peer computes eigenvalues/eigenvectors (with
     *      Armadillo functions), get the 2 Principal Components and
     *      save them in "subspace".
     * These operations are done for each dimension in UNCORR set.
    ***/
    auto cs_start = chrono::steady_clock::now();
    double ****subspace;
    subspace = (double ****) malloc(params.peers * sizeof(double ***));
    if (!subspace) {
        cerr << "Malloc error on subspace" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        subspace[peerID] = (double ***) malloc(uncorr_vars[peerID] * sizeof(double **));
        if (!subspace[peerID]) {
            cerr << "Malloc error on subspace for peer " << peerID << endl;
            exit(-1);
        }
        for (int m = 0; m < uncorr_vars[peerID]; ++m) {
            subspace[peerID][m] = (double **) malloc(2 * sizeof(double *));
            if (!subspace[peerID][m]) {
                cerr << "Malloc error on subspace for peer " << peerID << endl;
                exit(-1);
            }
            for (int k = 0; k < 2; ++k) {
                subspace[peerID][m][k] = (double *) malloc(partitionSize[peerID] * sizeof(double));
                if (!subspace[peerID][m][k]) {
                    cerr << "Malloc error on subspace for peer " << peerID << endl;
                    exit(-1);
                }
            }
        }
    }

    for (int subspaceID = 0; subspaceID < uncorr_vars[0]; ++subspaceID) {
        covar_storage = (double *) malloc(params.peers * 3 * 3 * sizeof(double));
        if (!covar_storage) {
            cerr << "Malloc error on covar_storage" << endl;
            exit(-1);
        }
        covar_i = (double **) malloc(params.peers * 3 * sizeof(double *));
        if (!covar_i) {
            cerr << "Malloc error on covar_i" << endl;
            exit(-1);
        }
        covar = (double ***) malloc(params.peers * sizeof(double **));
        if (!covar) {
            cerr << "Malloc error on covar" << endl;
            exit(-1);
        }
        for (int j = 0; j < params.peers * 3; ++j) {
            covar_i[j] = &covar_storage[j * 3];
        }

        for(int peerID = 0; peerID < params.peers; peerID++) {
            covar[peerID] = &covar_i[peerID * 3];

            for (int j = 0; j < partitionSize[peerID]; ++j) {
                if (peerID == 0) {
                    copyDimension(data, 2, uncorr[peerID][subspaceID], 0, peerLastItem[peerID], combine[peerID]);
                } else {
                    copyDimension(data, 2, uncorr[peerID][subspaceID], peerLastItem[peerID - 1] + 1, peerLastItem[peerID], combine[peerID]);
                }
            }
            computeLocalCovarianceMatrix(partitionSize[peerID], 3, combine[peerID], covar[peerID]);
        }

        int *cs_dims = (int *) malloc(params.peers * sizeof(int));
        if (!cs_dims) {
            cerr << "Malloc error on cs_dims" << endl;
            exit(-1);
        }
        fill_n(cs_dims, params.peers, 3);
        UDiagMatrixAverageConsensus(params, graph, cs_dims, covar);
        free(cs_dims);

        for(int peerID = 0; peerID < params.peers; peerID++) {
            computePCA(covar[peerID], combine[peerID], partitionSize[peerID], 3, subspace[peerID][subspaceID]);
        }
        free(covar);
        free(covar_i);
        free(covar_storage);
    }
    free(uncorr);
    free(combine);

    auto cs_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to create candidate subspaces: " <<
             chrono::duration_cast<chrono::nanoseconds>(cs_end - cs_start).count()*1e-9 << endl;
    }

    if (!outputOnFile) {
        printf("\nComputing distributed clustering...\n");
    }

    /***    Distributed K-Means
     * For each candidate subspace, peers cooperate to pick the optimal
     * number of clusters with the elbow criterion based on BetaCV metric.
     * The result of the clustering is saved in the structure "final".
     * 1) Peer 0 chooses a set of centroids randomly and broadcast them
     *      (this operation is done through an average consensus).
     * 2) All the peers start the distributed K-Means.
     *      Locally they compute the sum of all the coordinates in each cluster,
     *      the cardinality of each cluster coordinates' set and the local sum
     *      of squared errors. Then they save these informations in "localsum",
     *      "weights" and "error".
     * 3) An average consensus on the sum of localsum, weights and error is executed.
     * 4) The peers compute the new centroids and if the condition about the actual
     *      error and the previous error is satisfied, they stop the distributed
     *      K-Means with the number of cluster chosen for this run.
     * 5) They start the evaluation of BetaCV metric for the elbow criterion.
     *      If the condition about the elbow is satisfied, the distributed clustering
     *      for the actual candidate subspace is stopped.
    ***/
    auto clustering_start = chrono::steady_clock::now();

    cluster_report *final_i = (cluster_report *) calloc(uncorr_vars[0] * params.peers, sizeof(cluster_report));
    if (!final_i) {
        cerr << "Malloc error on final_i" << endl;
        exit(-1);
    }
    cluster_report **final = (cluster_report **) calloc(uncorr_vars[0], sizeof(cluster_report*));
    if (!final) {
        cerr << "Malloc error on final" << endl;
        exit(-1);
    }

    double *localsum_storage, **localsum_i, ***localsum, *weights_storage, **weights, *prev_err, *error;
    // Structures used for distributed BetaCV computation
    double *pts_incluster_storage, **pts_incluster, *c_mean_storage, **c_mean;

    for (int subspaceID = 0; subspaceID < uncorr_vars[0]; ++subspaceID) {
        final[subspaceID] = &final_i[subspaceID * params.peers];
        cluster_report *prev = (cluster_report *) calloc(params.peers, sizeof(cluster_report));
        if (!prev) {
            cerr << "Malloc error on prev" << endl;
            exit(-1);
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
                cerr << "Malloc error on localsum_storage" << endl;
                exit(-1);
            }
            localsum_i = (double **) malloc(2 * params.peers * sizeof(double *));
            if (!localsum_i) {
                cerr << "Malloc error on localsum_i" << endl;
                exit(-1);
            }
            localsum = (double ***) malloc(params.peers * sizeof(double **));
            if (!localsum) {
                cerr << "Malloc error on localsum" << endl;
                exit(-1);
            }
            weights_storage = (double *) malloc(params.peers * nCluster * sizeof(double));
            if (!weights_storage) {
                cerr << "Malloc error on weights_storage" << endl;
                exit(-1);
            }
            weights = (double **) malloc(params.peers * sizeof(double *));
            if (!weights) {
                cerr << "Malloc error on weights" << endl;
                exit(-1);
            }
            for (int i1 = 0; i1 < 2 * params.peers; ++i1) {
                localsum_i[i1] = &localsum_storage[i1 * nCluster];
            }
            for (int i1 = 0; i1 < params.peers; ++i1) {
                localsum[i1] = &localsum_i[i1 * 2];
                weights[i1] = &weights_storage[i1 * nCluster];
            }

            prev_err = (double *) malloc(params.peers * sizeof(double));
            if (!prev_err) {
                cerr << "Malloc error on prev_err" << endl;
                exit(-1);
            }
            error = (double *) malloc(params.peers * sizeof(double));
            if (!error) {
                cerr << "Malloc error on error" << endl;
                exit(-1);
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

                    computeLocalKMeans(partitionSize[peerID], centroids.slice(peerID), subspace[peerID][subspaceID], weights[peerID], localsum[peerID], error[peerID]);
                }

                LocalSumAverageConsensus(params, graph, nCluster, localsum);
                VectorAverageConsensus(params, graph, nCluster, weights);
                dimestimate = SimpleAverageConsensus(params, graph, error);

                for(int peerID = 0; peerID < params.peers; peerID++){
                    for (int l = 0; l < nCluster; ++l) {
                        centroids(0, l, peerID) = localsum[peerID][0][l] / weights[peerID][l];
                        centroids(1, l, peerID) = localsum[peerID][1][l] / weights[peerID][l];
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
            free(localsum_storage);
            free(localsum_i);
            free(localsum);
            free(weights_storage);
            free(weights);
            free(prev_err);
            free(error);

            for(int peerID = 0; peerID < params.peers; peerID++){
                if (nCluster == 1) {
                    final[subspaceID][peerID].centroids = centroids.slice(peerID);
                    final[subspaceID][peerID].k = nCluster;
                    final[subspaceID][peerID].BetaCV = 0.0;
                    final[subspaceID][peerID].cidx = (int *) calloc(partitionSize[peerID], sizeof(int));
                    if (!final[subspaceID][peerID].cidx) {
                        cerr << "Malloc error on final cidx for peer " << peerID << endl;
                        exit(-1);
                    }
                    prev[peerID].cidx = (int *) malloc(partitionSize[peerID] * sizeof(int));
                    if (!prev[peerID].cidx) {
                        cerr << "Malloc error on previous cidx for peer " << peerID << endl;
                        exit(-1);
                    }
                } else {
                    final[subspaceID][peerID].centroids = centroids.slice(peerID);
                    final[subspaceID][peerID].k = nCluster;
                    create_cidx_matrix(subspace[peerID][subspaceID], partitionSize[peerID], final[subspaceID][peerID]);
                }
            }
            /***    BetaCV Metric Computation
             * The peers need the intra-cluster (WCSS) and the inter-cluster (BCSS)
             * sum of squares, the number of distinct intra-cluster (N_in) and
             * inter-cluster (N_out) edges to compute BetaCV.
             * For N_in and N_out, the peers get the number of elements in each
             * cluster with an average consensus on the sum of "pts_incluster".
             * For BCSS, the peers get the mean of all the points  with an average
             * consensus on "c_mean".
             * Locally all the peers compute N_in, N_out, BCSS and the local estimate
             * of WCSS; then the general WCSS is reached with an average consensus on
             * the sum of WCSS.
             * After that each peer can compute the BetaCV metric locally and evaluates
             * the elbow criterion condition.
            ***/
            if (nCluster > 1) {
                pts_incluster_storage = (double *) calloc(params.peers * nCluster, sizeof(double));
                if (!pts_incluster_storage) {
                    cerr << "Malloc error on pts_incluster_storage" << endl;
                    exit(-1);
                }
                pts_incluster = (double **) malloc(params.peers * sizeof(double *));
                if (!pts_incluster) {
                    cerr << "Malloc error on pts_incluster" << endl;
                    exit(-1);
                }
                c_mean_storage = (double *) calloc(params.peers * 2, sizeof(double));
                if (!c_mean_storage) {
                    cerr << "Malloc error on c_mean_storage" << endl;
                    exit(-1);
                }
                c_mean = (double **) malloc(params.peers * sizeof(double *));
                if (!c_mean) {
                    cerr << "Malloc error on c_mean" << endl;
                    exit(-1);
                }
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    pts_incluster[peerID] = &pts_incluster_storage[peerID * nCluster];
                    c_mean[peerID] = &c_mean_storage[peerID * 2];

                    computeLocalC_Mean(subspace[peerID][subspaceID], partitionSize[peerID], c_mean[peerID]);
                    computeLocalPtsIncluster(partitionSize[peerID], pts_incluster[peerID], final[subspaceID][peerID]);
                }

                VectorAverageConsensus(params, graph, 2, c_mean);
                dimestimate = VectorAverageConsensus(params, graph, nCluster, pts_incluster);

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    for (int m = 0; m < nCluster; ++m) {
                        pts_incluster[peerID][m] = pts_incluster[peerID][m] / dimestimate[peerID];
                    }
                }

                double *bcss_storage = (double *) calloc(params.peers, sizeof(double));
                if (!bcss_storage) {
                    cerr << "Malloc error on bcss_storage" << endl;
                    exit(-1);
                }
                double *wcss_storage = (double *) calloc(params.peers, sizeof(double));
                if (!wcss_storage) {
                    cerr << "Malloc error on wcss_storage" << endl;
                    exit(-1);
                }
                double *nin_storage = (double *) calloc(params.peers, sizeof(double));
                if (!nin_storage) {
                    cerr << "Malloc error on nin_storage" << endl;
                    exit(-1);
                }
                double *nout_storage = (double *) calloc(params.peers, sizeof(double));
                if (!nout_storage) {
                    cerr << "Malloc error on nout_storage" << endl;
                    exit(-1);
                }
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    computeNin_Nout(nCluster, pts_incluster[peerID], nin_storage[peerID], nout_storage[peerID]);
                    bcss_storage[peerID] = computeBCSS(pts_incluster[peerID], final[subspaceID][peerID].centroids, c_mean[peerID]);
                    wcss_storage[peerID] = computeLocalWCSS(partitionSize[peerID], final[subspaceID][peerID], subspace[peerID][subspaceID]);
                }
                free(c_mean_storage);
                free(c_mean);
                free(pts_incluster_storage);
                free(pts_incluster);

                dimestimate = SimpleAverageConsensus(params, graph, wcss_storage);

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    wcss_storage[peerID] = wcss_storage[peerID] / dimestimate[peerID];
                    final[subspaceID][peerID].BetaCV = (nout_storage[peerID] * wcss_storage[peerID]) / (nin_storage[peerID] * bcss_storage[peerID]);
                }
                free(nout_storage);
                free(wcss_storage);
                free(nin_storage);
                free(bcss_storage);

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
        free(prev);
    }

    auto clustering_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to run K-Means: " <<
             chrono::duration_cast<chrono::nanoseconds>(clustering_end - clustering_start).count()*1e-9 << endl;
    }

    /***    Outliers Identification On Each Candidate Subspace
     * For each cluster:
     * 1) Each peer computes the local cluster size and saves it
     *      in "cluster_dim".
     * 2) The cardinality of the cluster is computed with an average
     *      consensus on "cluster_dim".
     * Then it starts the distributed outlier identification.
     * 1) Each peer computes the actual cluster dimension (cluster
     *      dimension without points discarded, considered as inliers
     *      in previous iterations) and saves it in "actual_cluster_dim"
     *      and the distance between the points and the centroids.
     * 2) These informations are exchanged with an average consensus and
     *      after that each peer can compute the radius of the circle used
     *      to discard points considered as inliers.
     * 3) Each peer evaluates the remaining points and if there are
     *      inliers, it updates "inliers".
     * 4) This information is exchanged with an average consensus on the sum
     *      of "inliers" and if one of the conditions on the inliers is satisfied,
     *      the outlier identification for the actual cluster is stopped.
    ***/
    auto identification_start = chrono::steady_clock::now();
    // Structure to keep record of inliers for each peer (for Outlier identification)
    bool ***discardedPts;
    discardedPts = (bool ***) malloc(params.peers * sizeof(bool **));
    if (!discardedPts) {
        cerr << "Malloc error on discardedPts" << endl;
        exit(-1);
    }
    for(int peerID = 0; peerID < params.peers; peerID++) {
        discardedPts[peerID] = (bool **) malloc(uncorr_vars[peerID] * sizeof(bool *));
        if (!discardedPts[peerID]) {
            cerr << "Malloc error on discardedPts for peer " << peerID << endl;
            exit(-1);
        }
        for (int m = 0; m < uncorr_vars[peerID]; ++m) {
            discardedPts[peerID][m] = (bool *) calloc(partitionSize[peerID], sizeof(bool));
            if (!discardedPts[peerID][m]) {
                cerr << "Malloc error on discardedPts for peer " << peerID << endl;
                exit(-1);
            }
        }
    }

    double *inliers, *prev_inliers, *cluster_dim, *actual_dist, *actual_cluster_dim;

    for (int i = 0; i < uncorr_vars[0]; ++i) {
        inliers = (double *) malloc(params.peers * sizeof(double));
        if (!inliers) {
            cerr << "Malloc error on inliers" << endl;
            exit(-1);
        }
        prev_inliers = (double *) malloc(params.peers * sizeof(double));
        if (!prev_inliers) {
            cerr << "Malloc error on prev_inliers" << endl;
            exit(-1);
        }
        cluster_dim = (double *) malloc(params.peers * sizeof(double));
        if (!cluster_dim) {
            cerr << "Malloc error on cluster_dim" << endl;
            exit(-1);
        }

        for (int l = 0; l < final[i][0].k; ++l) {
            for (int peerID = 0; peerID < params.peers; peerID++) {
                cluster_dim[peerID] = cluster_size(final[i][peerID], l, partitionSize[peerID]);
            }

            SimpleAverageConsensus(params, graph, cluster_dim);

            // Reset parameters for convergence estimate
            fill_n(inliers, params.peers, 0);
            Numberofconverged = params.peers;
            fill_n(converged, params.peers, false);
            actual_dist = (double *) calloc(params.peers, sizeof(double));
            if (!actual_dist) {
                cerr << "Malloc error on actual_dist" << endl;
                exit(-1);
            }
            actual_cluster_dim = (double *) calloc(params.peers, sizeof(double));
            if (!actual_cluster_dim) {
                cerr << "Malloc error on actual_cluster_dim" << endl;
                exit(-1);
            }

            while (Numberofconverged) {
                memcpy(prev_inliers, inliers, params.peers * sizeof(double));
                fill_n(actual_dist, params.peers, 0.0);
                fill_n(actual_cluster_dim, params.peers, 0.0);
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    if (converged[peerID])
                        continue;

                    actual_cluster_dim[peerID] = computeLocalClusterDimension(partitionSize[peerID], final[i][peerID],
                                                                              l, discardedPts[peerID][i]);
                    actual_dist[peerID] = computeLocalClusterDistance(partitionSize[peerID], final[i][peerID], l,
                                                                      discardedPts[peerID][i], subspace[peerID][i]);
                }

                SimpleAverageConsensus(params, graph, actual_dist);
                SimpleAverageConsensus(params, graph, actual_cluster_dim);

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    double dist_mean = actual_dist[peerID] / actual_cluster_dim[peerID];
                    inliers[peerID] += computeLocalInliers(partitionSize[peerID], final[i][peerID], l,
                                                           discardedPts[peerID][i], subspace[peerID][i],
                                                           dist_mean);
                }

                dimestimate = SimpleAverageConsensus(params, graph, inliers);

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
            free(actual_dist);
            free(actual_cluster_dim);
        }
        free(inliers);
        free(prev_inliers);
        free(cluster_dim);
    }

    /***    Final result broadcast to all the peers
     * Each peer sets "tot_num_data" to its partition size and then
     * all the peers estimate the total number of data with an average
     * consensus on the sum of "tot_num_data".
     * Each peer fills "global_outliers" with the local information, then
     * the global solution is reached with an average consensus on the sum
     * of each element in "global_outliers".
     * Finally peer 0 is queried to get the final result.
    ***/
    double *tot_num_data = (double *) calloc(params.peers, sizeof(double));
    if (!tot_num_data) {
        cerr << "Malloc error on tot_num_data" << endl;
        exit(-1);
    }
    for(int peerID = 0; peerID < params.peers; peerID++){
        tot_num_data[peerID] = partitionSize[peerID];
    }

    dimestimate = SimpleAverageConsensus(params, graph, tot_num_data);

    double **global_outliers = (double **) malloc(params.peers * sizeof(double *));
    if (!global_outliers) {
        cerr << "Malloc error on global_outliers" << endl;
        exit(-1);
    }

    for (int peerID = 0; peerID < params.peers; peerID++) {
        tot_num_data[peerID] = std::round(tot_num_data[peerID] / (double) dimestimate[peerID]);

        global_outliers[peerID] = (double *) calloc(tot_num_data[peerID], sizeof(double));
        if (!global_outliers[peerID]) {
            cerr << "Malloc error on global_outliers for peer" << peerID << endl;
            exit(-1);
        }

        if (peerID == 0) {
            getCountOutliersinSubspace(uncorr_vars[peerID], partitionSize[peerID], 0, discardedPts[peerID], global_outliers[peerID]);
        } else {
            int index = peerLastItem[peerID - 1] + 1;
            getCountOutliersinSubspace(uncorr_vars[peerID], partitionSize[peerID], index, discardedPts[peerID], global_outliers[peerID]);
        }
    }

    // Reset parameters for convergence estimate
    fill_n(dimestimate, params.peers, 0);
    dimestimate[0] = 1;
    Numberofconverged = params.peers;
    fill_n(converged, params.peers, false);

    double *prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        cerr << "Malloc error on prevestimate" << endl;
        exit(-1);
    }
    int *convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        cerr << "Malloc error on convRounds" << endl;
        exit(-1);
    }
    int rounds = 0;

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
            if(fanOut < neighborsSize){
                // randomly sample f adjacent vertices
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                for (int q = 0; q < tot_num_data[peerID]; ++q) {
                    computeAverage(&global_outliers[peerID][q], global_outliers[neighborID][q]);
                }
                memcpy(global_outliers[neighborID], global_outliers[peerID], tot_num_data[peerID] * sizeof(double));
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

    for (int peerID = 0; peerID < params.peers; peerID++) {
        for (int i = 0; i < tot_num_data[peerID]; ++i) {
            global_outliers[peerID][i] = std::round(global_outliers[peerID][i] / dimestimate[peerID]);
        }
    }

    cout << endl << "OUTLIERS:" << endl;
    for (int i = 0; i < tot_num_data[0]; ++i) {
        if (global_outliers[0][i] >= std::round(uncorr_vars[0] * params.percentageSubspaces)) {
            cout << i << ") ";
            for (int l = 0; l < n_dims; ++l) {
                cout << data[i][l] << " ";
            }
            cout << "(" << global_outliers[0][i] << ")" << endl;
        }
    }
    //data_out(subspace, peerLastItem, "iris.csv", discardedPts, params.peers, uncorr_vars[0], final[0]);
    free(tot_num_data);
    free(global_outliers);
    free(subspace);
    free(uncorr_vars);
    free(discardedPts);
    free(final);
    free(final_i);

    auto identification_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to identify outliers: " <<
             chrono::duration_cast<chrono::nanoseconds>(identification_end - identification_start).count()*1e-9 << endl;
    }

    igraph_destroy(&graph);

    free(data);
    free(data_storage);
    free(dimestimate);
    free(prevestimate);
    free(converged);
    free(convRounds);

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

int partitionData(int n_data, int peers, long *peerLastItem, long *partitionSize) {
    if (!peerLastItem || !partitionSize) {
        return NullPointerError(__FUNCTION__);
    }
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<> distr(-1, 1); // define the range

    for(int i = 0; i < peers - 1; i++){
        float rnd = distr(eng);
        //cerr << "rnd: " << rnd << "\n";
        long last_item = rnd * ((float)n_data/(float)peers) * 0.1 + (float) (i+1) * ((float)n_data/(float)peers) - 1;
        peerLastItem[i] = last_item;
    }
    peerLastItem[peers - 1] = n_data-1;

    /*** Check the partitioning correctness ***/
    long sum = peerLastItem[0] + 1;
    partitionSize[0] = peerLastItem[0] + 1;
    //cerr << "peer 0:" << sum << "\n";
    for(int i = 1; i < peers; i++) {
        sum += peerLastItem[i] - peerLastItem[i-1];
        partitionSize[i] = peerLastItem[i] - peerLastItem[i-1];
        //cerr << "peer " << i << ":" << peerLastItem[i] - peerLastItem[i-1] << "\n";
    }

    if(sum != n_data) {
        cout << "ERROR: n_data = " << n_data << "!= sum = " << sum << endl;
        return -3;
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
         << "number of consecutive rounds in which a peer must locally converge = "<< params.convLimit
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

bool isCorrDimension(int ndims, int firstDimension, double **pcc) {
    if (!pcc) {
        exit(NullPointerError(__FUNCTION__));
    }
    double overall = 0.0;
    for (int secondDimension = 0; secondDimension < ndims; ++secondDimension) {
        if (secondDimension != firstDimension) {
            overall += pcc[firstDimension][secondDimension];
        }
    }
    return ( (overall / ndims) >= 0 );
}

int computeCorrUncorrCardinality(double **pcc, int ndims, int &corr_vars, int &uncorr_vars) {
    if (!pcc || !(&corr_vars) || !(&uncorr_vars)) {
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

int copyDimension(double **data, int corr_vars, int dim, long start, long end, double **newstorage) {
    if (!data || !newstorage) {
        return NullPointerError(__FUNCTION__);
    }
    int elem = 0;
    for (int k = start; k <= end; ++k) {
        newstorage[corr_vars][elem] = data[k][dim];
        elem++;
    }
    return 0;
}

int computeLocalCovarianceMatrix(long partitionSize, int corr_vars, double **corrSet, double **covarianceMatrix) {
    if (!corrSet || !covarianceMatrix) {
        return NullPointerError(__FUNCTION__);
    }
    for (int i = 0; i < corr_vars; ++i) {
        for (int j = i; j < corr_vars; ++j) {
            covarianceMatrix[i][j] = 0;
            for (int k = 0; k < partitionSize; ++k) {
                covarianceMatrix[i][j] += corrSet[i][k] * corrSet[j][k];
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
    if (!subspace || !weights || !localsum || !(&error) || centroids.is_empty()) {
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
    if (!pts_incluster || !(&Nin) || !(&Nout)) {
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

double computeLocalWCSS(long partitionSize, cluster_report rep, double **subspace) {
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

int mergeSimpleMessages(int dim, double *peerElem, double *neighborElem) {
    if (!peerElem || !neighborElem) {
        return NullPointerError(__FUNCTION__);
    }
    for (int j = 0; j < dim; ++j) {
        computeAverage(&peerElem[j], neighborElem[j]);
    }
    memcpy(neighborElem, peerElem, dim * sizeof(double));
    return 0;
}

int mergeUDiagMatrix(int n_dims, double **UDiagMatrix_X, double **UDiagMatrix_Y) {
    if (!UDiagMatrix_X || !UDiagMatrix_Y) {
        return NullPointerError(__FUNCTION__);
    }

    for (int l = 0; l < n_dims; ++l) {
        for (int m = l; m < n_dims; ++m) {
            computeAverage(&UDiagMatrix_X[l][m], UDiagMatrix_Y[l][m]);
            UDiagMatrix_X[m][l] = UDiagMatrix_X[l][m];
        }
    }
    return 0;
}

double* SimpleAverageConsensus(Params params, igraph_t graph, double* structure) {
    if (!structure) {
        exit(NullPointerError(__FUNCTION__));
    }
    double *dimestimate = nullptr, *prevestimate = nullptr;
    bool *converged = nullptr;
    int *convRounds = nullptr;
    int Numberofconverged = params.peers;
    int rounds = 0;

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(memoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        memoryError(__FUNCTION__);
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

                computeAverage(&structure[peerID], structure[neighborID]);
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

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(memoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        memoryError(__FUNCTION__);
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

                mergeSimpleMessages(dim, structure[peerID], structure[neighborID]);
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

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(memoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        memoryError(__FUNCTION__);
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

                mergeUDiagMatrix(dim[peerID], structure[peerID], structure[peerID]);
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

    dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        exit(memoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        memoryError(__FUNCTION__);
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
                    computeAverage(&structure[peerID][0][l], structure[neighborID][0][l]);
                    computeAverage(&structure[peerID][1][l], structure[neighborID][1][l]);
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
        exit(memoryError(__FUNCTION__));
    }
    dimestimate[0] = 1;

    converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    fill_n(converged, params.peers, false);

    convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        memoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        memoryError(__FUNCTION__);
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
