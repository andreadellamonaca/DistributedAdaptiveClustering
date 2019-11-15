#include <iostream>
#include <igraph/igraph.h>
#include <cstring>
#include <random>
#include "adaptive_clustering.h"

/**
 * @file main.cpp
 */

/**
 * @struct Params
 * A structure containing parameters read from command-line.
 */
struct Params {
    int          peers; /**< The number of peers. */
    int          p_star; /**< Maximum number of peers. */
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
void computeLocalAverage(double **data, long partitionSize, int ndims, long start, long end, double *summaries) {
    double weight = 1 / (double) partitionSize;
    for (int i = start; i <= end; ++i) {
        for (int j = 0; j < ndims; ++j) {
            summaries[j] += weight * data[i][j];
        }
    }
};
void CenterData(double *summaries, int ndims, long start, long end, double **data) {
    for (int i = start; i <= end; ++i) {
        for (int j = 0; j < ndims; ++j) {
            data[i][j] -= summaries[j];
        }
    }
};
void computeLocalPCC(double **pcc, double *squaresum_dims, int ndims, long start, long end, double **data) {
    for (int l = 0; l < ndims; ++l) {
        pcc[l][l] = 1;
        for (int i = start; i <= end; ++i) {
            squaresum_dims[l] += pow(data[i][l], 2);
            for (int m = l + 1; m < ndims; ++m) {
                pcc[l][m] += data[i][l] * data[i][m];
            }
        }
    }
};
bool isCorrDimension(int ndims, int firstDimension, double **pcc) {
    double overall = 0.0;
    for (int secondDimension = 0; secondDimension < ndims; ++secondDimension) {
        if (secondDimension != firstDimension) {
            overall += pcc[firstDimension][secondDimension];
        }
    }
    return ( (overall / ndims) >= 0 );
};
void computeCorrUncorrCardinality(double **pcc, int ndims, int &corr_vars, int &uncorr_vars) {
    for (int i = 0; i < ndims; ++i) {
        if (isCorrDimension(ndims, i, pcc)) {
            corr_vars++;
        }
    }
    uncorr_vars = ndims - corr_vars;
};
void computeLocalCovarianceMatrix(long partitionSize, int corr_vars, double **corrSet, double **covarianceMatrix) {
    for (int i = 0; i < corr_vars; ++i) {
        for (int j = i; j < corr_vars; ++j) {
            covarianceMatrix[i][j] = 0;
            for (int k = 0; k < partitionSize; ++k) {
                covarianceMatrix[i][j] += corrSet[i][k] * corrSet[j][k];
            }
            covarianceMatrix[i][j] = covarianceMatrix[i][j] / (partitionSize - 1);
        }
    }
};
void computePCA(double **covarianceMatrix, double **corrSet, long partitionSize, int corr_vars, double **combine) {
    mat cov_mat(covarianceMatrix[0], corr_vars, corr_vars);
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, cov_mat);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < partitionSize; ++j) {
            double value = 0.0;
            for (int k = 0; k < corr_vars; ++k) {
                int col = corr_vars - i - 1;
                value += corrSet[k][j] * eigvec(k, col);
            }
            combine[i][j] = value;
        }
    }
};
void computeLocalKMeans(int nCluster, long partitionSize, mat centroids, double **subspace, double *weights, double **localsum, double &error) {
    for (int l = 0; l < nCluster; ++l) {
        weights[l] += 1;
        localsum[0][l] += centroids(0, l);
        localsum[1][l] += centroids(1, l);
    }
    for (int k = 0; k < partitionSize; ++k) {
        int clusterid = mindistCluster(centroids, nCluster, subspace[0][k], subspace[1][k]);

        weights[clusterid] += 1;
        localsum[0][clusterid] += subspace[0][k];
        localsum[1][clusterid] += subspace[1][k];
        error += pow(L2distance(centroids(0, clusterid), centroids(1, clusterid), subspace[0][k], subspace[1][k]), 2);
    }
};
void computeLocalMean_PtsIncluster(double **data, long partitionSize, double *summaries, long nCluster, double *pts_incluster, cluster_report rep) {
    double weight = 1 / (double) partitionSize;
    for (int k = 0; k < partitionSize; ++k) {
        for (int l = 0; l < 2; ++l) {
            summaries[l] += weight * data[l][k];
        }
        for (int m = 0; m < nCluster; ++m) {
            if (rep.cidx[k] == m) {
                pts_incluster[m]++;
            }
        }
    }
};
void computeNin_Nout(int nCluster, double *pts_incluster, double &Nin, double &Nout) {
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
};
double computeBCSS(int nCluster, double *pts_incluster, mat centroids, double *c_mean) {
    double bcss = 0;
    for (int m = 0; m < nCluster; ++m) {
        bcss += (pts_incluster[m] * L2distance(centroids(0, m), centroids(1, m), c_mean[0], c_mean[1]));
    }
    return bcss;
};
double computeLocalWCSS(int nCluster, double partitionSize, cluster_report rep, double **subspace) {
    double wcss = 0;
    for (int m = 0; m < nCluster; ++m) {
        for (int k = 0; k < partitionSize; ++k) {
            if (rep.cidx[k] == m) {
                wcss += L2distance(rep.centroids(0, m), rep.centroids(1, m), subspace[0][k], subspace[1][k]);
            }
        }
    }
    return wcss;
};
igraph_t generateGeometricGraph(igraph_integer_t n, igraph_real_t radius);
igraph_t generateBarabasiAlbertGraph(igraph_integer_t n, igraph_real_t power, igraph_integer_t m, igraph_real_t A);
igraph_t generateErdosRenyiGraph(igraph_integer_t n, igraph_erdos_renyi_t type, igraph_real_t param);
igraph_t generateRegularGraph(igraph_integer_t n, igraph_integer_t k);
igraph_t generateRandomGraph(int type, int n);
void printGraphType(int type);

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
    int p_star = -1;
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
        } else if (strcmp(argv[i], "-ps") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing maximum number of peers parameter." << endl;
                return -1;
            }
            p_star = stoi(argv[i]);
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
        } else if (strcmp(argv[i], "-pspace") == 0) {
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
    partitionSize = (long *) calloc(peers, sizeof(long));
    if (!partitionSize) {
        cerr << "Malloc error on partitionSize" << endl;
        exit(-1);
    }
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
        exit(EXIT_FAILURE);
    }

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
    if (p_star == -1)
        p_star = peers;

    params.p_star = p_star;
    outputOnFile = params.outputFilename.size() > 0;

    if (!outputOnFile) {
        printf("\n\nPARAMETERS:\n");
        cout << "input file= " << params.inputFilename << "\n";
        cout << "percentage in circle = " << params.percentageIncircle << "\n";
        cout << "elbow threshold = " << params.elbowThreshold << "\n";
        cout << "convergence clustering threshold = " << params.convClusteringThreshold << "\n";
        cout << "percentage subspaces = " << params.percentageSubspaces << "\n";
        cout << "k_max = " << params.k_max << "\n";
        cout << "peers = " << params.peers << "\n";
        cout << "fan-out = " << params.fanOut << "\n";
        cout << "graph type = ";
        printGraphType(params.graphType);
        cout << "local convergence tolerance = "<< params.convThreshold << "\n";
        cout << "number of consecutive rounds in which a peer must locally converge = "<< params.convLimit << "\n";
        cout << "\n\n";
    }

    /*** Graph generation ***/
    // turn on attribute handling in igraph
    igraph_i_set_attribute_table(&igraph_cattribute_table);

    // seed igraph PRNG
    igraph_rng_seed(igraph_rng_default(), 42);

    auto gengraphstart = chrono::steady_clock::now();
    // generate a connected random graph
    graph = generateRandomGraph(params.graphType, params.peers);

    auto gengraphend = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to generate the random graph: " <<
             chrono::duration_cast<chrono::nanoseconds>(gengraphend - gengraphstart).count()*1e-9 << endl;
    }

    // determine minimum and maximum vertex degree for the graph
    igraph_vector_t result;
    igraph_real_t mindeg;
    igraph_real_t maxdeg;

    igraph_vector_init(&result, 0);
    igraph_degree(&graph, &result, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
    igraph_vector_minmax(&result, &mindeg, &maxdeg);
    if (!outputOnFile) {
        cout << "Minimum degree is " << (int) mindeg << ", Maximum degree is " << (int) maxdeg << endl;
    }

    // this is used to estimate the number of peers
    double *dimestimate = (double *) calloc(params.peers, sizeof(double));
    if (!dimestimate) {
        cerr << "Malloc error on dimestimate" << endl;
        exit(-1);
    }
    dimestimate[0] = 1;

    int Numberofconverged = params.peers;
    bool *converged = (bool *) calloc(params.peers, sizeof(bool));
    if (!converged) {
        cerr << "Malloc error on converged" << endl;
        exit(-1);
    }
    fill_n(converged, params.peers, false);

    int *convRounds = (int *) calloc(params.peers, sizeof(int));
    if (!convRounds) {
        cerr << "Malloc error on convRounds" << endl;
        exit(-1);
    }
    int rounds = 0;

    double *prevestimate = (double *) calloc(params.peers, sizeof(double));
    if (!prevestimate) {
        cerr << "Malloc error on prevestimate" << endl;
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
    for (int i = 0; i < params.peers; ++i) {
        avgsummaries[i] = &avg_storage[i * n_dims];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        if (peerID == 0) {
            computeLocalAverage(data, partitionSize[peerID], n_dims, 0, peerLastItem[peerID], avgsummaries[peerID]);
        } else {
            computeLocalAverage(data, partitionSize[peerID], n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID], avgsummaries[peerID]);
        }
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

                for (int j = 0; j < n_dims; ++j) {
                    avgsummaries[peerID][j] = (avgsummaries[peerID][j] + avgsummaries[neighborID][j]) / 2;
                }
                memcpy(avgsummaries[neighborID], avgsummaries[peerID], n_dims * sizeof(double));
                double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                dimestimate[peerID] = mean;
                dimestimate[neighborID] = mean;
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
    for (int i = 0; i < params.peers; ++i) {
        pcc[i] = &pcc_i[i * n_dims];
        squaresum_dims[i] = &squaresum_dims_storage[i * n_dims];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        if (peerID == 0) {
            computeLocalPCC(pcc[peerID], squaresum_dims[peerID], n_dims, 0, peerLastItem[peerID], data);
        } else {
            computeLocalPCC(pcc[peerID], squaresum_dims[peerID], n_dims, peerLastItem[peerID-1] + 1, peerLastItem[peerID], data);
        }
    }

    // Reset parameters for convergence estimate
    fill_n(dimestimate, params.peers, 0);
    dimestimate[0] = 1;
    Numberofconverged = params.peers;
    fill_n(converged, params.peers, false);
    fill_n(convRounds, params.peers, 0);
    rounds = 0;

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

                for (int l = 0; l < n_dims; ++l) {
                    squaresum_dims[peerID][l] = (squaresum_dims[peerID][l] + squaresum_dims[neighborID][l]) / 2;
                    for (int m = l + 1; m < n_dims; ++m) {
                        pcc[peerID][l][m] = (pcc[peerID][l][m] + pcc[neighborID][l][m]) / 2;
                    }
                }
                memcpy(squaresum_dims[neighborID], squaresum_dims[peerID], n_dims * sizeof(double));
                memcpy(pcc[neighborID][0], pcc[peerID][0], n_dims * n_dims * sizeof(double));
                double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                dimestimate[peerID] = mean;
                dimestimate[neighborID] = mean;
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

    for(int peerID = 0; peerID < params.peers; peerID++){
        for (int l = 0; l < n_dims; ++l) {
            for (int m = l + 1; m < n_dims; ++m) {
                pcc[peerID][l][m] = pcc[peerID][l][m] / sqrt(squaresum_dims[peerID][l] * squaresum_dims[peerID][m]);
                pcc[peerID][m][l] = pcc[peerID][l][m];
            }
        }
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
                int elem = 0;
                if (peerID == 0) {
                    for (int k = 0; k <= peerLastItem[peerID]; ++k) {
                        corr[peerID][corr_vars[peerID]][elem] = data[k][i];
                        elem++;
                    }
                } else {
                    for (int k = peerLastItem[peerID-1] + 1; k <= peerLastItem[peerID]; ++k) {
                        corr[peerID][corr_vars[peerID]][elem] = data[k][i];
                        elem++;
                    }
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

    // Reset parameters for convergence estimate
    fill_n(dimestimate, params.peers, 0);
    dimestimate[0] = 1;
    Numberofconverged = params.peers;
    fill_n(converged, params.peers, false);
    fill_n(convRounds, params.peers, 0);
    rounds = 0;

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

                for (int l = 0; l < corr_vars[peerID]; ++l) {
                    for (int k = l; k < corr_vars[peerID]; ++k) {
                        covar[peerID][l][k] = (covar[peerID][l][k] + covar[neighborID][l][k]) / 2;
                        covar[peerID][k][l] = covar[peerID][l][k];
                    }
                }
                memcpy(covar[neighborID][0], covar[peerID][0], corr_vars[peerID] * corr_vars[peerID] * sizeof(double));
                double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                dimestimate[peerID] = mean;
                dimestimate[neighborID] = mean;
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

    for (int m = 0; m < uncorr_vars[0]; ++m) {
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
        for (int j = 0; j < params.peers; ++j) {
            covar[j] = &covar_i[j * 3];
        }

        for(int peerID = 0; peerID < params.peers; peerID++) {
            for (int j = 0; j < partitionSize[peerID]; ++j) {
                if (peerID == 0) {
                    combine[peerID][2][j] = data[j][uncorr[peerID][m]];
                } else {
                    int index = peerLastItem[peerID-1] + 1 + j;
                    combine[peerID][2][j] = data[index][uncorr[peerID][m]];
                }
            }

            computeLocalCovarianceMatrix(partitionSize[peerID], 3, combine[peerID], covar[peerID]);
        }

        // Reset parameters for convergence estimate
        fill_n(dimestimate, params.peers, 0);
        dimestimate[0] = 1;
        Numberofconverged = params.peers;
        fill_n(converged, params.peers, false);
        fill_n(convRounds, params.peers, 0);
        rounds = 0;

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

                    for (int l = 0; l < 3; ++l) {
                        for (int k = l; k < 3; ++k) {
                            covar[peerID][l][k] = (covar[peerID][l][k] + covar[neighborID][l][k]) / 2;
                            covar[peerID][k][l] = covar[peerID][l][k];
                        }
                    }
                    memcpy(covar[neighborID][0], covar[peerID][0], 3 * 3 * sizeof(double));
                    double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                    dimestimate[peerID] = mean;
                    dimestimate[neighborID] = mean;
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

        for(int peerID = 0; peerID < params.peers; peerID++) {
            computePCA(covar[peerID], combine[peerID], partitionSize[peerID], 3, subspace[peerID][m]);
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

    for (int i = 0; i < uncorr_vars[0]; ++i) {
        final[i] = &final_i[i * params.peers];
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
            // Reset parameters for convergence estimate
            fill_n(dimestimate, params.peers, 0);
            dimestimate[0] = 1;
            Numberofconverged = params.peers;
            fill_n(converged, params.peers, false);
            fill_n(convRounds, params.peers, 0);
            rounds = 0;
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
                    for(int i1 = 0; i1 < neighborsSize; i1++){
                        int neighborID = (int) VECTOR(neighbors)[i1];
                        igraph_integer_t edgeID;
                        igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                        centroids.slice(peerID) = (centroids.slice(peerID) + centroids.slice(neighborID)) / 2;
                        centroids.slice(neighborID) = centroids.slice(peerID);
                        double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                        dimestimate[peerID] = mean;
                        dimestimate[neighborID] = mean;
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

                    computeLocalKMeans(nCluster, partitionSize[peerID], centroids.slice(peerID), subspace[peerID][i], weights[peerID], localsum[peerID], error[peerID]);
                }

                // Reset parameters for convergence estimate
                fill_n(dimestimate, params.peers, 0);
                dimestimate[0] = 1;
                int N_converged = params.peers;
                bool *converged2 = (bool *) calloc(params.peers, sizeof(bool));
                if (!converged2) {
                    cerr << "Malloc error on converged2" << endl;
                    exit(-1);
                }
                int *convRounds2 = (int *) calloc(params.peers, sizeof(int));
                if (!convRounds2) {
                    cerr << "Malloc error on convRounds2" << endl;
                    exit(-1);
                }
                int rounds2 = 0;

                while( (params.roundsToExecute < 0 && N_converged) || params.roundsToExecute > 0){
                    memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
                    for(int peerID = 0; peerID < params.peers; peerID++){
                        // check peer convergence
                        if(params.roundsToExecute < 0 && converged2[peerID])
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
                        for(int i1 = 0; i1 < neighborsSize; i1++){
                            int neighborID = (int) VECTOR(neighbors)[i1];
                            igraph_integer_t edgeID;
                            igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                            for (int l = 0; l < nCluster; ++l) {
                                localsum[peerID][0][l] = (localsum[peerID][0][l] + localsum[neighborID][0][l]) / 2;
                                localsum[peerID][1][l] = (localsum[peerID][1][l] + localsum[neighborID][1][l]) / 2;
                                weights[peerID][l] = (weights[peerID][l] + weights[neighborID][l]) / 2;
                            }
                            memcpy(localsum[neighborID][0], localsum[peerID][0], 2 * nCluster * sizeof(double));
                            memcpy(weights[neighborID], weights[peerID], nCluster * sizeof(double));
                            double mean_error = (error[peerID] + error[neighborID]) / 2;
                            error[peerID] = mean_error;
                            error[neighborID] = mean_error;
                            double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                            dimestimate[peerID] = mean;
                            dimestimate[neighborID] = mean;
                        }
                        igraph_vector_destroy(&neighbors);
                    }

                    // check local convergence
                    if (params.roundsToExecute < 0) {
                        for(int peerID = 0; peerID < params.peers; peerID++){
                            if(converged2[peerID])
                                continue;
                            bool dimestimateconv;
                            if(prevestimate[peerID])
                                dimestimateconv = fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) < params.convThreshold;
                            else
                                dimestimateconv = false;

                            if(dimestimateconv)
                                convRounds2[peerID]++;
                            else
                                convRounds2[peerID] = 0;

                            converged2[peerID] = (convRounds2[peerID] >= params.convLimit);
                            if(converged2[peerID]){
                                N_converged --;
                            }
                        }
                    }
                    rounds2++;
                    //cerr << "\r Active peers: " << N_converged << " - Rounds: " << rounds2 << "          ";
                    params.roundsToExecute--;
                }
                free(converged2);
                free(convRounds2);

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
                    final[i][peerID].centroids = centroids.slice(peerID);
                    final[i][peerID].k = nCluster;
                    final[i][peerID].BetaCV = 0.0;
                    final[i][peerID].cidx = (int *) calloc(partitionSize[peerID], sizeof(int));
                    if (!final[i][peerID].cidx) {
                        cerr << "Malloc error on final cidx for peer " << peerID << endl;
                        exit(-1);
                    }
                    prev[peerID].cidx = (int *) malloc(partitionSize[peerID] * sizeof(int));
                    if (!prev[peerID].cidx) {
                        cerr << "Malloc error on previous cidx for peer " << peerID << endl;
                        exit(-1);
                    }
                } else {
                    final[i][peerID].centroids = centroids.slice(peerID);
                    final[i][peerID].k = nCluster;
                    create_cidx_matrix(subspace[peerID][i], partitionSize[peerID], final[i][peerID]);
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
                    computeLocalMean_PtsIncluster(subspace[peerID][i], partitionSize[peerID], c_mean[peerID], nCluster, pts_incluster[peerID], final[i][peerID]);
                }

                // Reset parameters for convergence estimate
                fill_n(dimestimate, params.peers, 0);
                dimestimate[0] = 1;
                Numberofconverged = params.peers;
                fill_n(converged, params.peers, false);
                fill_n(convRounds, params.peers, 0);
                rounds = 0;

                while ((params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0) {
                    memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
                    for (int peerID = 0; peerID < params.peers; peerID++) {
                        // check peer convergence
                        if (params.roundsToExecute < 0 && converged[peerID])
                            continue;
                        // determine peer neighbors
                        igraph_vector_t neighbors;
                        igraph_vector_init(&neighbors, 0);
                        igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
                        long neighborsSize = igraph_vector_size(&neighbors);
                        if (fanOut < neighborsSize) {
                            // randomly sample f adjacent vertices
                            igraph_vector_shuffle(&neighbors);
                            igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
                        }

                        neighborsSize = igraph_vector_size(&neighbors);
                        for (int k = 0; k < neighborsSize; k++) {
                            int neighborID = (int) VECTOR(neighbors)[k];
                            igraph_integer_t edgeID;
                            igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                            for (int l = 0; l < 2; ++l) {
                                c_mean[peerID][l] = (c_mean[peerID][l] + c_mean[neighborID][l]) / 2;
                            }
                            for (int m = 0; m < nCluster; ++m) {
                                pts_incluster[peerID][m] = (pts_incluster[peerID][m] + pts_incluster[neighborID][m]) / 2;
                            }
                            memcpy(c_mean[neighborID], c_mean[peerID], 2 * sizeof(double));
                            memcpy(pts_incluster[neighborID], pts_incluster[peerID], nCluster * sizeof(double));
                            double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                            dimestimate[peerID] = mean;
                            dimestimate[neighborID] = mean;
                        }
                        igraph_vector_destroy(&neighbors);
                    }

                    // check local convergence
                    if (params.roundsToExecute < 0) {
                        for (int peerID = 0; peerID < params.peers; peerID++) {
                            if (converged[peerID])
                                continue;
                            bool dimestimateconv;
                            if (prevestimate[peerID])
                                dimestimateconv =
                                        fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) <
                                        params.convThreshold;
                            else
                                dimestimateconv = false;

                            if (dimestimateconv)
                                convRounds[peerID]++;
                            else
                                convRounds[peerID] = 0;

                            converged[peerID] = (convRounds[peerID] >= params.convLimit);
                            if (converged[peerID]) {
                                Numberofconverged--;
                            }
                        }
                    }
                    rounds++;
                    //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
                    params.roundsToExecute--;
                }

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
                    bcss_storage[peerID] = computeBCSS(nCluster, pts_incluster[peerID], final[i][peerID].centroids, c_mean[peerID]);
                    wcss_storage[peerID] = computeLocalWCSS(nCluster, partitionSize[peerID], final[i][peerID], subspace[peerID][i]);
                }
                free(c_mean_storage);
                free(c_mean);
                free(pts_incluster_storage);
                free(pts_incluster);

                // Reset parameters for convergence estimate
                fill_n(dimestimate, params.peers, 0);
                dimestimate[0] = 1;
                Numberofconverged = params.peers;
                fill_n(converged, params.peers, false);
                fill_n(convRounds, params.peers, 0);
                rounds = 0;

                while ((params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0) {
                    memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
                    for (int peerID = 0; peerID < params.peers; peerID++) {
                        // check peer convergence
                        if (params.roundsToExecute < 0 && converged[peerID])
                            continue;
                        // determine peer neighbors
                        igraph_vector_t neighbors;
                        igraph_vector_init(&neighbors, 0);
                        igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
                        long neighborsSize = igraph_vector_size(&neighbors);
                        if (fanOut < neighborsSize) {
                            // randomly sample f adjacent vertices
                            igraph_vector_shuffle(&neighbors);
                            igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
                        }

                        neighborsSize = igraph_vector_size(&neighbors);
                        for (int k = 0; k < neighborsSize; k++) {
                            int neighborID = (int) VECTOR(neighbors)[k];
                            igraph_integer_t edgeID;
                            igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                            double wcss_mean = (wcss_storage[peerID] + wcss_storage[neighborID]) / 2;
                            wcss_storage[peerID] = wcss_mean;
                            wcss_storage[neighborID] = wcss_mean;
                            double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                            dimestimate[peerID] = mean;
                            dimestimate[neighborID] = mean;
                        }
                        igraph_vector_destroy(&neighbors);
                    }

                    // check local convergence
                    if (params.roundsToExecute < 0) {
                        for (int peerID = 0; peerID < params.peers; peerID++) {
                            if (converged[peerID])
                                continue;
                            bool dimestimateconv;
                            if (prevestimate[peerID])
                                dimestimateconv =
                                        fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) <
                                        params.convThreshold;
                            else
                                dimestimateconv = false;

                            if (dimestimateconv)
                                convRounds[peerID]++;
                            else
                                convRounds[peerID] = 0;

                            converged[peerID] = (convRounds[peerID] >= params.convLimit);
                            if (converged[peerID]) {
                                Numberofconverged--;
                            }
                        }
                    }
                    rounds++;
                    //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
                    params.roundsToExecute--;
                }
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    wcss_storage[peerID] = wcss_storage[peerID] / dimestimate[peerID];
                    final[i][peerID].BetaCV = (nout_storage[peerID] * wcss_storage[peerID]) / (nin_storage[peerID] * bcss_storage[peerID]);
                }
                free(nout_storage);
                free(wcss_storage);
                free(nin_storage);
                free(bcss_storage);

                if (fabs(prev[0].BetaCV - final[i][0].BetaCV) <= params.elbowThreshold) {
                    cout << "The optimal K is " << final[i][0].k << endl;
                    break;
                } else {
                    for (int peerID = 0; peerID < params.peers; peerID++) {
                        prev[peerID] = final[i][peerID];
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
    bool ***incircle;
    incircle = (bool ***) malloc(params.peers * sizeof(bool **));
    if (!incircle) {
        cerr << "Malloc error on incircle" << endl;
        exit(-1);
    }
    for(int peerID = 0; peerID < params.peers; peerID++) {
        incircle[peerID] = (bool **) malloc(uncorr_vars[peerID] * sizeof(bool *));
        if (!incircle[peerID]) {
            cerr << "Malloc error on incircle for peer " << peerID << endl;
            exit(-1);
        }
        for (int m = 0; m < uncorr_vars[peerID]; ++m) {
            incircle[peerID][m] = (bool *) calloc(partitionSize[peerID], sizeof(bool));
            if (!incircle[peerID][m]) {
                cerr << "Malloc error on incircle for peer " << peerID << endl;
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

            // Reset parameters for convergence estimate
            fill_n(dimestimate, params.peers, 0);
            dimestimate[0] = 1;
            Numberofconverged = params.peers;
            fill_n(converged, params.peers, false);
            fill_n(convRounds, params.peers, 0);
            rounds = 0;

            while ((params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0) {
                memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    // check peer convergence
                    if (params.roundsToExecute < 0 && converged[peerID])
                        continue;
                    // determine peer neighbors
                    igraph_vector_t neighbors;
                    igraph_vector_init(&neighbors, 0);
                    igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
                    long neighborsSize = igraph_vector_size(&neighbors);
                    if (fanOut < neighborsSize) {
                        // randomly sample f adjacent vertices
                        igraph_vector_shuffle(&neighbors);
                        igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
                    }

                    neighborsSize = igraph_vector_size(&neighbors);
                    for (int k = 0; k < neighborsSize; k++) {
                        int neighborID = (int) VECTOR(neighbors)[k];
                        igraph_integer_t edgeID;
                        igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                        double cluster_dim_mean = (cluster_dim[peerID] + cluster_dim[neighborID]) / 2;
                        cluster_dim[peerID] = cluster_dim_mean;
                        cluster_dim[neighborID] = cluster_dim_mean;
                        double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                        dimestimate[peerID] = mean;
                        dimestimate[neighborID] = mean;
                    }
                    igraph_vector_destroy(&neighbors);
                }

                // check local convergence
                if (params.roundsToExecute < 0) {
                    for (int peerID = 0; peerID < params.peers; peerID++) {
                        if (converged[peerID])
                            continue;
                        bool dimestimateconv;
                        if (prevestimate[peerID])
                            dimestimateconv =
                                    fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) <
                                    params.convThreshold;
                        else
                            dimestimateconv = false;

                        if (dimestimateconv)
                            convRounds[peerID]++;
                        else
                            convRounds[peerID] = 0;

                        converged[peerID] = (convRounds[peerID] >= params.convLimit);
                        if (converged[peerID]) {
                            Numberofconverged--;
                        }
                    }
                }
                rounds++;
                //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
                params.roundsToExecute--;
            }

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

                    for (int k = 0; k < partitionSize[peerID]; ++k) {
                        if (final[i][peerID].cidx[k] == l && !(incircle[peerID][i][k]) ) {
                            actual_dist[peerID] += L2distance(final[i][peerID].centroids.at(0, l), final[i][peerID].centroids.at(1, l), subspace[peerID][i][0][k], subspace[peerID][i][1][k]);
                            actual_cluster_dim[peerID]++;
                        }
                    }
                }

                // Reset parameters for convergence estimate
                fill_n(dimestimate, params.peers, 0);
                dimestimate[0] = 1;
                int N_converged = params.peers;
                bool *converged2 = (bool *) calloc(params.peers, sizeof(bool));
                if (!converged2) {
                    cerr << "Malloc error on converged2" << endl;
                    exit(-1);
                }
                fill_n(convRounds, params.peers, 0);
                rounds = 0;

                while ((params.roundsToExecute < 0 && N_converged) || params.roundsToExecute > 0) {
                    memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
                    for (int peerID = 0; peerID < params.peers; peerID++) {
                        // check peer convergence
                        if (params.roundsToExecute < 0 && converged2[peerID])
                            continue;
                        // determine peer neighbors
                        igraph_vector_t neighbors;
                        igraph_vector_init(&neighbors, 0);
                        igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
                        long neighborsSize = igraph_vector_size(&neighbors);
                        if (fanOut < neighborsSize) {
                            // randomly sample f adjacent vertices
                            igraph_vector_shuffle(&neighbors);
                            igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
                        }

                        neighborsSize = igraph_vector_size(&neighbors);
                        for (int k = 0; k < neighborsSize; k++) {
                            int neighborID = (int) VECTOR(neighbors)[k];
                            igraph_integer_t edgeID;
                            igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                            double dist_mean = (actual_dist[peerID] + actual_dist[neighborID]) / 2;
                            double cluster_dim_mean = (actual_cluster_dim[peerID] + actual_cluster_dim[neighborID]) / 2;
                            actual_dist[peerID] = dist_mean;
                            actual_dist[neighborID] = dist_mean;
                            actual_cluster_dim[peerID] = cluster_dim_mean;
                            actual_cluster_dim[neighborID] = cluster_dim_mean;

                            double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                            dimestimate[peerID] = mean;
                            dimestimate[neighborID] = mean;
                        }
                        igraph_vector_destroy(&neighbors);
                    }

                    // check local convergence
                    if (params.roundsToExecute < 0) {
                        for (int peerID = 0; peerID < params.peers; peerID++) {
                            if (converged2[peerID])
                                continue;
                            bool dimestimateconv;
                            if (prevestimate[peerID])
                                dimestimateconv =
                                        fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) <
                                        params.convThreshold;
                            else
                                dimestimateconv = false;

                            if (dimestimateconv)
                                convRounds[peerID]++;
                            else
                                convRounds[peerID] = 0;

                            converged2[peerID] = (convRounds[peerID] >= params.convLimit);
                            if (converged2[peerID]) {
                                N_converged--;
                            }
                        }
                    }
                    rounds++;
                    //cerr << "\r Active peers: " << N_converged << " - Rounds: " << rounds << "          ";
                    params.roundsToExecute--;
                }

                for (int peerID = 0; peerID < params.peers; peerID++) {
                    double dist_mean = actual_dist[peerID] / actual_cluster_dim[peerID];
                    for (int k = 0; k < partitionSize[peerID]; ++k) {
                        if (final[i][peerID].cidx[k] == l && !(incircle[peerID][i][k]) ) {
                            if (L2distance(final[i][peerID].centroids.at(0, l), final[i][peerID].centroids.at(1, l), subspace[peerID][i][0][k], subspace[peerID][i][1][k])
                                <= dist_mean) {
                                incircle[peerID][i][k] = true;
                                inliers[peerID]++;
                            }
                        }
                    }
                }

                // Reset parameters for convergence estimate
                fill_n(dimestimate, params.peers, 0);
                dimestimate[0] = 1;
                N_converged = params.peers;
                fill_n(converged2, params.peers, false);
                fill_n(convRounds, params.peers, 0);
                rounds = 0;

                while ((params.roundsToExecute < 0 && N_converged) || params.roundsToExecute > 0) {
                    memcpy(prevestimate, dimestimate, params.peers * sizeof(double));
                    for (int peerID = 0; peerID < params.peers; peerID++) {
                        // check peer convergence
                        if (params.roundsToExecute < 0 && converged2[peerID])
                            continue;
                        // determine peer neighbors
                        igraph_vector_t neighbors;
                        igraph_vector_init(&neighbors, 0);
                        igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
                        long neighborsSize = igraph_vector_size(&neighbors);
                        if (fanOut < neighborsSize) {
                            // randomly sample f adjacent vertices
                            igraph_vector_shuffle(&neighbors);
                            igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
                        }

                        neighborsSize = igraph_vector_size(&neighbors);
                        for (int k = 0; k < neighborsSize; k++) {
                            int neighborID = (int) VECTOR(neighbors)[k];
                            igraph_integer_t edgeID;
                            igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                            double inliers_mean = (inliers[peerID] + inliers[neighborID]) / 2;
                            inliers[peerID] = inliers_mean;
                            inliers[neighborID] = inliers_mean;

                            double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                            dimestimate[peerID] = mean;
                            dimestimate[neighborID] = mean;
                        }
                        igraph_vector_destroy(&neighbors);
                    }

                    // check local convergence
                    if (params.roundsToExecute < 0) {
                        for (int peerID = 0; peerID < params.peers; peerID++) {
                            if (converged2[peerID])
                                continue;
                            bool dimestimateconv;
                            if (prevestimate[peerID])
                                dimestimateconv =
                                        fabs((prevestimate[peerID] - dimestimate[peerID]) / prevestimate[peerID]) <
                                        params.convThreshold;
                            else
                                dimestimateconv = false;

                            if (dimestimateconv)
                                convRounds[peerID]++;
                            else
                                convRounds[peerID] = 0;

                            converged2[peerID] = (convRounds[peerID] >= params.convLimit);
                            if (converged2[peerID]) {
                                N_converged--;
                            }
                        }
                    }
                    rounds++;
                    //cerr << "\r Active peers: " << N_converged << " - Rounds: " << rounds << "          ";
                    params.roundsToExecute--;
                }
                free(converged2);

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

    // Reset parameters for convergence estimate
    fill_n(dimestimate, params.peers, 0);
    dimestimate[0] = 1;
    Numberofconverged = params.peers;
    fill_n(converged, params.peers, false);
    fill_n(convRounds, params.peers, 0);
    rounds = 0;

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

                double num_data_mean = (tot_num_data[peerID] + tot_num_data[neighborID]) / 2;
                tot_num_data[peerID] = num_data_mean;
                tot_num_data[neighborID] = num_data_mean;
                double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                dimestimate[peerID] = mean;
                dimestimate[neighborID] = mean;
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

    double **global_outliers = (double **) malloc(params.peers * sizeof(double *));
    if (!global_outliers) {
        cerr << "Malloc error on global_outliers" << endl;
        exit(-1);
    }

    for (int peerID = 0; peerID < params.peers; peerID++) {
        tot_num_data[peerID] = tot_num_data[peerID] / (double) dimestimate[peerID];
        int size = std::round(tot_num_data[peerID]);
        global_outliers[peerID] = (double *) calloc(size, sizeof(double));
        if (!global_outliers[peerID]) {
            cerr << "Malloc error on global_outliers for peer" << peerID << endl;
            exit(-1);
        }

        for (int k = 0; k < partitionSize[peerID]; ++k) {
            for (int j = 0; j < uncorr_vars[peerID]; ++j) {
                if (!incircle[peerID][j][k]) {
                    int index;
                    if (peerID == 0) {
                        index = 0;
                    } else {
                        index = peerLastItem[peerID - 1] + 1;
                    }
                    global_outliers[peerID][index+k]++;
                }
            }
        }
    }

    // Reset parameters for convergence estimate
    fill_n(dimestimate, params.peers, 0);
    dimestimate[0] = 1;
    Numberofconverged = params.peers;
    fill_n(converged, params.peers, false);
    fill_n(convRounds, params.peers, 0);
    rounds = 0;

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

                int size = std::round(tot_num_data[peerID]);
                for (int q = 0; q < size; ++q) {
                    global_outliers[peerID][q] = (global_outliers[peerID][q] + global_outliers[neighborID][q]) / 2;
                }
                memcpy(global_outliers[neighborID], global_outliers[peerID], size * sizeof(double));
                double mean = (dimestimate[peerID] + dimestimate[neighborID]) / 2;
                dimestimate[peerID] = mean;
                dimestimate[neighborID] = mean;
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
        for (int q = 0; q < std::round(tot_num_data[peerID]); ++q) {
            global_outliers[peerID][q] = std::round(global_outliers[peerID][q] / dimestimate[peerID]);
        }
    }

    cout << endl << "OUTLIERS:" << endl;
    for (int q = 0; q < std::round(tot_num_data[0]); ++q) {
        if (global_outliers[0][q] >= std::round(uncorr_vars[0] * params.percentageSubspaces)) {
            cout << q << ") ";
            for (int l = 0; l < n_dims; ++l) {
                cout << data[q][l] << " ";
            }
            cout << "(" << global_outliers[0][q] << ")" << endl;
        }
    }
    //data_out(subspace, peerLastItem, "iris.csv", incircle, params.peers, uncorr_vars[0], final[0]);
    free(tot_num_data);
    free(global_outliers);
    free(subspace);
    free(uncorr_vars);
    free(incircle);
    free(final);
    free(final_i);

    auto identification_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to identify outliers: " <<
             chrono::duration_cast<chrono::nanoseconds>(identification_end - identification_start).count()*1e-9 << endl;
    }

    igraph_vector_destroy(&result);
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
            << "-p          number of peers\n"
            << "-f          fan-out of peers\n"
            << "-s          seed\n"
            << "-d          graph type: 1 geometric 2 Barabasi-Albert 3 Erdos-Renyi 4 regular\n"
            << "-ct         convergence tolerance\n"
            << "-cl         number of consecutive rounds in which convergence must be satisfied\n"
            << "-of         output filename, if specified a file with this name containing all of the peers stats is written\n"
            << "-k          max number of clusters to try in elbow criterion\n"
            << "-et         threshold for the selection of optimal number of clusters in Elbow method\n"
            << "-clst       the local convergence tolerance for distributed K-Means.\n"
            << "-pi         percentage of points in a cluster to be evaluated as inlier\n"
            << "-pspace     percentage of subspaces in which a point must be outlier to be evaluated as general outlier\n"
            << "-if         input filename\n"
            << "-as         enable autoseeding\n\n";
}

igraph_t generateGeometricGraph(igraph_integer_t n, igraph_real_t radius)
{
    igraph_t G_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the geometric model
    igraph_grg_game(&G_graph, n, radius, 0, 0, 0);

    igraph_is_connected(&G_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&G_graph);
        igraph_grg_game(&G_graph, n, radius, 0, 0, 0);

        igraph_is_connected(&G_graph, &connected, IGRAPH_WEAK);
    }
    return G_graph;
}

igraph_t generateBarabasiAlbertGraph(igraph_integer_t n, igraph_real_t power, igraph_integer_t m, igraph_real_t A)
{
    // n = The number of vertices in the graph
    // power = Power of the preferential attachment. The probability that a vertex is cited is proportional to d^power+A, where d is its degree, power and A are given by arguments. In the classic preferential attachment model power=1
    // m = number of outgoing edges generated for each vertex
    // A = The probability that a vertex is cited is proportional to d^power+A, where d is its degree, power and A are given by arguments

    igraph_t BA_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the Barabasi-Albert model
    igraph_barabasi_game(/* graph=    */ &BA_graph,
            /* n=        */ n,
            /* power=    */ power,
            /* m=        */ m,
            /* outseq=   */ 0,
            /* outpref=  */ 0,
            /* A=        */ A,
            /* directed= */ IGRAPH_UNDIRECTED,
            /* algo=     */ IGRAPH_BARABASI_PSUMTREE,
            /* start_from= */ 0);

    igraph_is_connected(&BA_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&BA_graph);
        igraph_barabasi_game(/* graph=    */ &BA_graph,
                /* n=        */ n,
                /* power=    */ power,
                /* m=        */ m,
                /* outseq=   */ 0,
                /* outpref=  */ 0,
                /* A=        */ A,
                /* directed= */ IGRAPH_UNDIRECTED,
                /* algo=     */ IGRAPH_BARABASI_PSUMTREE,
                /* start_from= */ 0);

        igraph_is_connected(&BA_graph, &connected, IGRAPH_WEAK);
    }
    return BA_graph;
}

igraph_t generateErdosRenyiGraph(igraph_integer_t n, igraph_erdos_renyi_t type, igraph_real_t param)
{
    // n = The number of vertices in the graph
    // type = IGRAPH_ERDOS_RENYI_GNM G(n,m) graph, m edges are selected uniformly randomly in a graph with n vertices.
    //      = IGRAPH_ERDOS_RENYI_GNP G(n,p) graph, every possible edge is included in the graph with probability p

    igraph_t ER_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the Erdos-Renyi model
    igraph_erdos_renyi_game(&ER_graph, type, n, param, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);

    igraph_is_connected(&ER_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&ER_graph);
        igraph_erdos_renyi_game(&ER_graph, type, n, param, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);

        igraph_is_connected(&ER_graph, &connected, IGRAPH_WEAK);
    }
    return ER_graph;
}




igraph_t generateRegularGraph(igraph_integer_t n, igraph_integer_t k)
{
    // n = The number of vertices in the graph
    // k = The degree of each vertex in an undirected graph. For undirected graphs, at least one of k and the number of vertices must be even.

    igraph_t R_graph;
    igraph_bool_t connected;

    // generate a connected regular random graph
    igraph_k_regular_game(&R_graph, n, k, IGRAPH_UNDIRECTED, 0);

    igraph_is_connected(&R_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&R_graph);
        igraph_k_regular_game(&R_graph, n, k, IGRAPH_UNDIRECTED, 0);

        igraph_is_connected(&R_graph, &connected, IGRAPH_WEAK);
    }
    return R_graph;
}



igraph_t generateRandomGraph(int type, int n)
{
    igraph_t random_graph;

    switch (type) {
        case 1:
            random_graph = generateGeometricGraph(n, sqrt(100.0/(float)n));
            break;
        case 2:
            random_graph = generateBarabasiAlbertGraph(n, 1.0, 5, 1.0);
            break;
        case 3:
            random_graph = generateErdosRenyiGraph(n, IGRAPH_ERDOS_RENYI_GNP, 10.0/(float)n);
            // random_graph = generateErdosRenyiGraph(n, IGRAPH_ERDOS_RENYI_GNM, ceil(n^2/3));
            break;
        case 4:
            random_graph = generateRegularGraph(n, n-1);
            break;
        default:
            break;
    }
    return random_graph;
}


void printGraphType(int type)
{
    switch (type) {
        case 1:
            printf("Geometric random graph\n");
            break;
        case 2:
            printf("Barabasi-Albert random graph\n");
            break;
        case 3:
            printf("Erdos-Renyi random graph\n");
            break;
        case 4:
            printf("Regular random graph\n");
            break;
        default:
            break;
    }
}