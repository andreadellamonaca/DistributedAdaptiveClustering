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
igraph_t generateGeometricGraph(igraph_integer_t n, igraph_real_t radius);
igraph_t generateBarabasiAlbertGraph(igraph_integer_t n, igraph_real_t power, igraph_integer_t m, igraph_real_t A);
igraph_t generateErdosRenyiGraph(igraph_integer_t n, igraph_erdos_renyi_t type, igraph_real_t param);
igraph_t generateRegularGraph(igraph_integer_t n, igraph_integer_t k);
igraph_t generateRandomGraph(int type, int n);
void printGraphType(int type);

int main(int argc, char **argv) {

    int n_dims; // number of dimensions
    int n_data; // number of data
    long *peerLastItem; // index of a peer last item
    long *partitionSize; // size of a peer partition
    int peers = 10; // number of peers
    int fanOut = 3; //fan-out of peers
    //uint32_t seed = 16033099; // seed for the PRNG
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

    /*** Parse command-line parameters ***/
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
            elbowThreshold = stof(argv[i]);
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
    double **data, *data_storage;
    getDatasetDims(inputFilename, &n_dims, &n_data);

    data_storage = (double *) malloc(n_dims * n_data * sizeof(double));
    if (!data_storage) {
        cerr << "Malloc error on data_storage" << endl;
        exit(-1);
    }
    data = (double **) malloc(n_data * sizeof(double *));
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

     ***/
    auto std_start = chrono::steady_clock::now();

    /*** Local Average for standardization ***/
    double **avgsummaries, *avg_storage;
    avg_storage = (double *) malloc(params.peers * n_dims * sizeof(double));
    if (!avg_storage) {
        cerr << "Malloc error on avg_storage" << endl;
        exit(-1);
    }
    fill_n(avg_storage, params.peers * n_dims, 0);
    avgsummaries = (double **) malloc(params.peers * sizeof(double *));
    if (!avgsummaries) {
        cerr << "Malloc error on avgsummaries" << endl;
        exit(-1);
    }

    for (int i = 0; i < params.peers; ++i) {
        avgsummaries[i] = &avg_storage[i * n_dims];
    }
    /*** Local estimate ***/
    for(int peerID = 0; peerID < params.peers; peerID++){
        double weight = 1 / (double) partitionSize[peerID];
        if (peerID == 0) {
            for (int i = 0; i <= peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    avgsummaries[peerID][j] += weight * data[i][j];
                }
            }
        } else {
            for (int i = peerLastItem[peerID-1] + 1; i <= peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    avgsummaries[peerID][j] += weight * data[i][j];
                }
            }
        }
    }
    /*** Consensus on average ***/
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
    /*** Standardization ***/
    for(int peerID = 0; peerID < params.peers; peerID++){
        if (peerID == 0) {
            for (int i = 0; i <= peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    data[i][j] -= avgsummaries[peerID][j];
                }
            }
        } else {
            for (int i = peerLastItem[peerID-1] + 1; i <= peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    data[i][j] -= avgsummaries[peerID][j];
                }
            }
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

    auto pcc_start = chrono::steady_clock::now();
    /*** Pearson Matrix structure ***/
    double *pcc_storage, **pcc_i, ***pcc;
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
    double **squaresum_dims, *squaresum_dims_storage;
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
        for (int l = 0; l < n_dims; ++l) {
            /*** Local sum of squares of each dimension (for Pearson coefficient denominator) ***/
            pcc[peerID][l][l] = 1;
            if (peerID == 0) {
                for (int i = 0; i <= peerLastItem[peerID]; ++i) {
                    squaresum_dims[peerID][l] += pow(data[i][l], 2);
                    for (int m = l + 1; m < n_dims; ++m) {
                        pcc[peerID][l][m] += data[i][l] * data[i][m];
                    }
                }
            } else {
                for (int i = peerLastItem[peerID-1] + 1; i <= peerLastItem[peerID]; ++i) {
                    squaresum_dims[peerID][l] += pow(data[i][l], 2);
                    for (int m = l + 1; m < n_dims; ++m) {
                        pcc[peerID][l][m] += data[i][l] * data[i][m];
                    }
                }
            }
        }
    }
    /*** Consensus on Pearson matrix ***/
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

    auto partition_start = chrono::steady_clock::now();
    /*** Structure for CORR and UNCORR cardinality ***/
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

    for(int peerID = 0; peerID < params.peers; peerID++) {
        for (int i = 0; i < n_dims; ++i) {
            double overall = 0.0;
            for (int j = 0; j < n_dims; ++j) {
                if (j != i) {
                    overall += pcc[peerID][i][j];
                }
            }
            if ((overall / n_dims) >= 0) {
                corr_vars[peerID]++;
            } else {
                uncorr_vars[peerID]++;
            }
        }

        if (peerID == 0) {
            cout << "Correlated dimensions: " << corr_vars[peerID] << ", " << "Uncorrelated dimensions: "
                 << uncorr_vars[peerID] << endl;
            if (corr_vars[peerID] < 2) {
                cerr << "Correlated dimensions must be more than 1 in order to apply PCA!" << endl;
                exit(-1);
            }
            if (uncorr_vars[peerID] == 0) {
                cerr << "There are no candidate subspaces!" << endl;
                exit(-1);
            }
        }
    }
    /*** Structures for CORR and UNCORR partitioning ***/
    int *uncorr_storage, **uncorr;
    uncorr_storage = (int *) malloc(params.peers * uncorr_vars[0] * sizeof(int));
    if (!uncorr_storage) {
        cerr << "Malloc error on uncorr_storage" << endl;
        exit(-1);
    }
    uncorr = (int **) (malloc(params.peers * sizeof(int *)));
    if (!uncorr) {
        cerr << "Malloc error on corr or uncorr" << endl;
        exit(-1);
    }
    for (int i = 0; i < params.peers; ++i) {
        uncorr[i] = &uncorr_storage[i * uncorr_vars[0]];
    }
    double ***corr;
    corr = (double ***) malloc(params.peers * sizeof(double **));
    if (!corr) {
        cerr << "Malloc error on corr" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
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

        corr_vars[peerID] = 0, uncorr_vars[peerID] = 0;
        for (int i = 0; i < n_dims; ++i) {
            double overall = 0.0;
            for (int j = 0; j < n_dims; ++j) {
                if (i != j) {
                    overall += pcc[peerID][i][j];
                }
            }
            if ((overall/n_dims) >= 0) {
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

    auto pca_start = chrono::steady_clock::now();

    /*** Structure to compute Covariance Matrix for CORR set ***/
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
    /*** Local estimate for Covariance Matrix for CORR ***/
    for(int peerID = 0; peerID < params.peers; peerID++){
        for (int i = 0; i < corr_vars[peerID]; ++i) {
            for (int j = i; j < corr_vars[peerID]; ++j) {
                covar[peerID][i][j] = 0;
                for (int k = 0; k < partitionSize[peerID]; ++k) {
                    covar[peerID][i][j] += corr[peerID][i][k] * corr[peerID][j][k];
                }
                covar[peerID][i][j] = covar[peerID][i][j] / (partitionSize[peerID] - 1);
            }
        }
    }
    /*** Consensus on Covariance Matrix for CORR ***/
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
    /*** Structure to save PC1_corr, PC2_corr and the i-th dimension of UNCORR set ***/
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
        /*** Each peer compute eigenvalues/eigenvectors locally, get the 2 Principal Components and save in "combine" ***/
        mat cov_mat(covar[peerID][0], corr_vars[peerID], corr_vars[peerID]);
        vec eigval;
        mat eigvec;
        eig_sym(eigval, eigvec, cov_mat);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < partitionSize[peerID]; ++j) {
                double value = 0.0;
                for (int k = 0; k < corr_vars[peerID]; ++k) {
                    int col = corr_vars[peerID] - i - 1;
                    value += corr[peerID][k][j] * eigvec(k, col);
                }
                combine[peerID][i][j] = value;
            }
        }
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

    auto cs_start = chrono::steady_clock::now();

    /*** Structure to save the generated subspaces ***/
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
            /*** Save the m-th dimension of UNCORR set in "combine" ***/
            for (int j = 0; j < partitionSize[peerID]; ++j) {
                if (peerID == 0) {
                    combine[peerID][2][j] = data[j][uncorr[peerID][m]];
                } else {
                    int index = peerLastItem[peerID-1] + 1 + j;
                    combine[peerID][2][j] = data[index][uncorr[peerID][m]];
                }
            }
            /*** Compute Covariance Matrix on PC1_corr, PC2_corr and the m-th dimension of UNCORR set ***/
            for (int l = 0; l < 3; ++l) {
                for (int j = l; j < 3; ++j) {
                    covar[peerID][l][j] = 0;
                    for (int k = 0; k < partitionSize[peerID]; ++k) {
                        covar[peerID][l][j] += combine[peerID][l][k] * combine[peerID][j][k];
                    }
                    covar[peerID][l][j] = covar[peerID][l][j] / (partitionSize[peerID] - 1);
                }
            }
        }
        /*** Consensus on Covariance Matrix ***/
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
        /*** Each peer computes the eigenvalues/eigenvectors, get the 2 Principal Components and save in "subspace" ***/
        for(int peerID = 0; peerID < params.peers; peerID++) {
            mat cov_mat(covar[peerID][0], 3, 3);
            vec eigval;
            mat eigvec;
            eig_sym(eigval, eigvec, cov_mat);

            for (int j = 0; j < partitionSize[peerID]; ++j) {
                for (int i = 0; i < 2; ++i) {
                    double value = 0.0;
                    for (int k = 0; k < 3; ++k) {
                        int col = 3 - i - 1;
                        value += combine[peerID][k][j] * eigvec(k, col);
                    }
                    subspace[peerID][m][i][j] = value;
                }
            }
        }
        free(covar);
        free(covar_i);
        free(covar_storage);
    }
    free(uncorr);
    free(uncorr_storage);
    free(combine);

    auto cs_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to create candidate subspaces: " <<
             chrono::duration_cast<chrono::nanoseconds>(cs_end - cs_start).count()*1e-9 << endl;
    }

    if (!outputOnFile) {
        printf("\nComputing distributed clustering...\n");
    }

    auto clustering_start = chrono::steady_clock::now();
    /*** Structure to keep record of inliers ***/
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
    /*** Structure to save the cluster report for each peer ***/
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
    /*** Structures used for distributed clustering ***/
    double *localsum_storage, **localsum_i, ***localsum, *weights_storage, **weights, *prev_err, *error;
    /*** Structures used for distributed BetaCV computation ***/
    double *pts_storage, **pts_incluster, *c_mean_storage, **c_mean;

    for (int i = 0; i < uncorr_vars[0]; ++i) {
        final[i] = &final_i[i * params.peers];
        cluster_report *prev = (cluster_report *) calloc(params.peers, sizeof(cluster_report));
        if (!prev) {
            cerr << "Malloc error on prev" << endl;
            exit(-1);
        }

        for (int j = 1; j <= params.k_max; ++j) {
            cube centroids(2, j, params.peers, fill::zeros);
            /*** Peer 0 set random centroids ***/
            centroids.slice(0) = randu<mat>(2, j);
            /*** Broadcast random centroids ***/
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
            /*** Start distributed clustering ***/
            localsum_storage = (double *) malloc(j * 2 * params.peers * sizeof(double));
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
            weights_storage = (double *) malloc(params.peers * j * sizeof(double));
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
                localsum_i[i1] = &localsum_storage[i1 * j];
            }
            for (int i1 = 0; i1 < params.peers; ++i1) {
                localsum[i1] = &localsum_i[i1 * 2];
                weights[i1] = &weights_storage[i1 * j];
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
                fill_n(weights_storage, j * params.peers, 0.0);
                fill_n(localsum_storage, 2 * j * params.peers, 0.0);
                fill_n(error, params.peers, 0.0);
                for(int peerID = 0; peerID < params.peers; peerID++){
                    // check peer convergence
                    if(converged[peerID])
                        continue;
                    /*** Local K-means clustering ***/
                    for (int l = 0; l < j; ++l) {
                        weights[peerID][l] += 1;
                        localsum[peerID][0][l] += centroids(0, l, peerID);
                        localsum[peerID][1][l] += centroids(1, l, peerID);
                    }
                    for (int k = 0; k < partitionSize[peerID]; ++k) {
                        int clusterid = 0;
                        double mindist = pow(L2distance(centroids(0, 0, peerID), centroids(1, 0, peerID), subspace[peerID][i][0][k], subspace[peerID][i][1][k]), 2);
                        for (int l = 1; l < j; ++l) {
                            double dist = pow(L2distance(centroids(0, l, peerID), centroids(1, l, peerID), subspace[peerID][i][0][k], subspace[peerID][i][1][k]), 2);
                            if ( dist < mindist ) {
                                mindist = dist;
                                clusterid = l;
                            }
                        }
                        weights[peerID][clusterid] += 1;
                        localsum[peerID][0][clusterid] += subspace[peerID][i][0][k];
                        localsum[peerID][1][clusterid] += subspace[peerID][i][1][k];
                        error[peerID] += pow(L2distance(centroids(0, clusterid, peerID), centroids(1, clusterid, peerID), subspace[peerID][i][0][k], subspace[peerID][i][1][k]), 2);
                    }
                }
                /*** Consensus on centroids ***/
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

                            for (int l = 0; l < j; ++l) {
                                localsum[peerID][0][l] = (localsum[peerID][0][l] + localsum[neighborID][0][l]) / 2;
                                localsum[peerID][1][l] = (localsum[peerID][1][l] + localsum[neighborID][1][l]) / 2;
                                weights[peerID][l] = (weights[peerID][l] + weights[neighborID][l]) / 2;
                            }
                            memcpy(localsum[neighborID][0], localsum[peerID][0], 2 * j * sizeof(double));
                            memcpy(weights[neighborID], weights[peerID], j * sizeof(double));
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
                    for (int l = 0; l < j; ++l) {
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
                if (j == 1) {
                    final[i][peerID].centroids = centroids.slice(peerID);
                    final[i][peerID].k = j;
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
                    final[i][peerID].k = j;
                    create_cidx_matrix(subspace[peerID][i], partitionSize[peerID], final[i][peerID]);
                }
            }
            /*** Start distributed BetaCV computation (WCSS, BCSS, N_in and N_out) ***/
            if (j > 1) {
                pts_storage = (double *) calloc(params.peers * j, sizeof(double));
                if (!pts_storage) {
                    cerr << "Malloc error on pts_storage" << endl;
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
                    pts_incluster[peerID] = &pts_storage[peerID * j];
                    c_mean[peerID] = &c_mean_storage[peerID * 2];
                    double weight = 1 / (double) partitionSize[peerID];
                    for (int k = 0; k < partitionSize[peerID]; ++k) {
                        for (int l = 0; l < 2; ++l) {
                            c_mean[peerID][l] += weight * subspace[peerID][i][l][k];
                        }
                        for (int m = 0; m < j; ++m) {
                            if (final[i][peerID].cidx[k] == m) {
                                pts_incluster[peerID][m]++;
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
                            for (int m = 0; m < j; ++m) {
                                pts_incluster[peerID][m] = (pts_incluster[peerID][m] + pts_incluster[neighborID][m]) / 2;
                            }
                            memcpy(c_mean[neighborID], c_mean[peerID], 2 * sizeof(double));
                            memcpy(pts_incluster[neighborID], pts_incluster[peerID], j * sizeof(double));
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
                    for (int m = 0; m < j; ++m) {
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
                    for (int m = 0; m < j; ++m) {
                        nin_storage[peerID] += pts_incluster[peerID][m] * (pts_incluster[peerID][m] - 1);
                        for (int k = 0; k < j; ++k) {
                            if (k != m) {
                                nout_storage[peerID] += pts_incluster[peerID][m] * pts_incluster[peerID][k];
                            }
                        }
                        bcss_storage[peerID] += (pts_incluster[peerID][m] * L2distance(final[i][peerID].centroids.at(0, m), final[i][peerID].centroids.at(1, m), c_mean[peerID][0], c_mean[peerID][1]));
                        for (int k = 0; k < partitionSize[peerID]; ++k) {
                            if (final[i][peerID].cidx[k] == m) {
                                wcss_storage[peerID] += L2distance(final[i][peerID].centroids.at(0, m), final[i][peerID].centroids.at(1, m), subspace[peerID][i][0][k], subspace[peerID][i][1][k]);
                            }
                        }
                    }
                    nin_storage[peerID] = nin_storage[peerID] / 2;
                    nout_storage[peerID] = nout_storage[peerID] / 2;
                }
                free(c_mean_storage);
                free(c_mean);
                free(pts_storage);
                free(pts_incluster);
                /*** Consensus on WCSS ***/
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

                            wcss_storage[peerID] = (wcss_storage[peerID] + wcss_storage[neighborID]) / 2;
                            memcpy(&wcss_storage[neighborID], &wcss_storage[peerID], sizeof(double));
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
                    double metric = (nout_storage[peerID] * wcss_storage[peerID]) / (nin_storage[peerID] * bcss_storage[peerID]);
                    final[i][peerID].BetaCV = metric;
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
        /*** Structures for inliers and cluster dimension for outliers identification ***/
        double *inliers = (double *) malloc(params.peers * sizeof(double));
        if (!inliers) {
            cerr << "Malloc error on inliers" << endl;
            exit(-1);
        }
        double *prev_inliers = (double *) malloc(params.peers * sizeof(double));
        if (!prev_inliers) {
            cerr << "Malloc error on prev_inliers" << endl;
            exit(-1);
        }
        double *cluster_dim = (double *) malloc(params.peers * sizeof(double));
        if (!cluster_dim) {
            cerr << "Malloc error on cluster_dim" << endl;
            exit(-1);
        }

        for (int l = 0; l < final[i][0].k; ++l) {
            /*** Local computation of cardinality for each cluster ***/
            for (int peerID = 0; peerID < params.peers; peerID++) {
                cluster_dim[peerID] = cluster_size(final[i][peerID], l, partitionSize[peerID]);
            }
            /*** Consensus on cardinality for each cluster ***/
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
            double *actual_dist = (double *) calloc(params.peers, sizeof(double));
            if (!actual_dist) {
                cerr << "Malloc error on actual_dist" << endl;
                exit(-1);
            }
            double *actual_cluster_dim = (double *) calloc(params.peers, sizeof(double));
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

    cout << "------------------OUTLIERS---------------------" << endl;
    for (int peerID = 0; peerID < params.peers; peerID++) {
        cout << "peer " << peerID << endl;
        for (int k = 0; k < partitionSize[peerID]; ++k) {
            int occurrence = 0;
            for (int j = 0; j < uncorr_vars[peerID]; ++j) {
                if (!incircle[peerID][j][k]) {
                    occurrence++;
                }
            }

            if (occurrence >= std::round(uncorr_vars[peerID] * params.percentageSubspaces)) {
                int index;
                if (peerID == 0) {
                    index = 0;
                } else {
                    index = peerLastItem[peerID - 1] + 1;
                }
                cout << index+k << ") ";
                for (int l = 0; l < n_dims; ++l) {
                    cout << data[index+k][l] << " ";
                }
                cout << "(" << occurrence << ")\n";
            }
        }
    }
    data_out(subspace, peerLastItem, "iris.csv", incircle, params.peers, uncorr_vars[0], final[0]);
    free(subspace);
    free(uncorr_vars);
    free(incircle);
    free(final);
    free(final_i);

    auto clustering_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to run K-Means and identify outliers: " <<
             chrono::duration_cast<chrono::nanoseconds>(clustering_end - clustering_start).count()*1e-9 << endl;
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