#include <iostream>
#include <igraph/igraph.h>
#include <cstring>
#include <random>
#include "adaptive_clustering.h"

using namespace std;

struct Params {
    int          peers;
    int          p_star;
    string       inputFilename;
    string       outputFilename;
    double       convThreshold;
    int          convLimit;
    int          graphType;
    int          fanOut;
    int          roundsToExecute;
    double       delta;
    long         k_max;
    double       elbow_thr;
    double       percentage_incircle;
    double       percentage_subspaces;
};

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
    int peers = 10; // number of peers
    int fanOut = 3; //fan-out of peers
    uint32_t seed = 16033099; // seed for the PRNG
    int graphType = 2; // graph distribution: 1 geometric 2 Barabasi-Albert 3 Erdos-Renyi 4 regular (clique)
    double convThreshold = 0.001; // local convergence tolerance
    int convLimit = 3; // number of consecutive rounds in which a peer must locally converge
    int roundsToExecute = -1;
    int p_star = -1;
    double delta = 0.04;
    long k_max = 10;
    double elbow_thr = 0.02;
    double percentage_incircle = 0.9;
    double percentage_subspaces = 0.8;

    Params          params;
    double          elapsed;
    int             iterations;
    bool            autoseed = false;
    bool            outputOnFile = false;
    string          inputFilename = "../datasets/Iris.csv";
    string          outputFilename;
    igraph_t        graph;

    /*** parse command-line parameters ***/
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-delta") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing delta parameter." << endl;
                return -1;
            }
            delta = stod(argv[i]);
        } else if (strcmp(argv[i], "-p") == 0) {
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
        } else if (strcmp(argv[i], "-ethr") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of threshold for Elbow method.\n";
                return -1;
            }
            elbow_thr = stof(argv[i]);
        } else if (strcmp(argv[i], "-incircle") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of inlier points.\n";
                return -1;
            }
            percentage_incircle = stof(argv[i]);
        } else if (strcmp(argv[i], "-subspace") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of subspace in which an outlier must be.\n";
                return -1;
            }
            percentage_subspaces = stof(argv[i]);
        } else if (strcmp(argv[i], "-if") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing input file name.\n";
                return -1;
            }
            inputFilename = string(argv[i]);
        }else if (strcmp(argv[i], "-as") == 0) {
            autoseed = true;
        } else {
            usage(argv[0]);
            return -1;
        }
    }

    /*** dataset reading, loading and standardization ***/
    double **data, *data_storage;
    getDatasetDims(inputFilename, &n_dims, &n_data);

    data_storage = (double *) malloc(n_dims * n_data * sizeof(double));
    if (data_storage == nullptr) {
        cout << "Malloc error on data_storage" << endl;
        exit(-1);
    }
    data = (double **) malloc(n_data * sizeof(double *));
    if (data == nullptr) {
        cout << "Malloc error on data" << endl;
        exit(-1);
    }

    for (int i = 0; i < n_data; ++i) {
        data[i] = &data_storage[i * n_dims];
    }

    loadData(inputFilename, data, n_dims);

    /*** Compute last item for each peer***/
    peerLastItem = (long *) calloc(peers, sizeof(long));
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<> distr(-1, 1); // define the range

    /*** Partitioning phase ***/
    for(int i = 0; i < peers - 1; i++){
        float rnd = distr(eng);
        //cerr << "rnd: " << rnd << "\n";
        long last_item = rnd * ((float)n_data/(float)peers) * 0.1 + (float) (i+1) * ((float)n_data/(float)peers) - 1;
        peerLastItem[i] = last_item;
    }

    peerLastItem[peers - 1] = n_data-1;

    /*** check the partitioning correctness ***/
    long sum = peerLastItem[0] + 1;
    //cerr << "peer 0:" << sum << "\n";
    for(int i = 1; i < peers; i++) {
        sum += peerLastItem[i] - peerLastItem[i-1];
        //cerr << "peer " << i << ":" << peerLastItem[i] - peerLastItem[i-1] << "\n";
    }

    if(sum != n_data) {
        cout << "ERROR: n_data = " << n_data << "!= sum = " << sum << endl;
        exit(EXIT_FAILURE);
    }

    /*** assign parameters read from command line ***/
    params.peers = peers;
    params.fanOut = fanOut;
    params.graphType = graphType;
    params.convThreshold = convThreshold;
    params.convLimit = convLimit;
    params.outputFilename = outputFilename;
    params.roundsToExecute = roundsToExecute;
    params.delta = delta;
    params.elbow_thr = elbow_thr;
    params.k_max = k_max;
    params.percentage_incircle = percentage_incircle;
    params.percentage_subspaces = percentage_subspaces;
    params.inputFilename = inputFilename;
    if (p_star == -1)
        p_star = peers;

    params.p_star = p_star;

    outputOnFile = params.outputFilename.size() > 0;
/*
    if (!outputOnFile) {
        printf("\n\nPARAMETERS:\n");
        cout << "input file= " << params.inputFilename << "\n";
        cout << "percentage in circle = " << params.percentage_incircle << "\n";
        cout << "elbow threshold = " << params.elbow_thr << "\n";
        cout << "percentage subspaces = " << params.percentage_subspaces << "\n";
        cout << "k_max = " << params.k_max << "\n";
        cout << "peers = " << params.peers << "\n";
        cout << "fan-out = " << params.fanOut << "\n";
        cout << "graph type = ";
        printGraphType(params.graphType);
        cout << "local convergence tolerance = "<< params.convThreshold << "\n";
        cout << "number of consecutive rounds in which a peer must locally converge = "<< params.convLimit << "\n";
        cout << "\n\n";
    }
*/
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
    dimestimate[0] = 1;

    int Numberofconverged = params.peers;
    bool *converged = (bool *) calloc(params.peers, sizeof(bool));
    fill_n(converged, params.peers, false);

    int *convRounds = (int *) calloc(params.peers, sizeof(int));
    int rounds = 0;

    double *prevestimate = (double *) calloc(params.peers, sizeof(double));

    // Apply Dataset Standardization to each peer' substream
    if (!outputOnFile) {
        printf("\nApplying Dataset Standardization to each peer' substream...\n");
    }

    auto std_start = chrono::steady_clock::now();

    //Local Average
    double **avgsummaries, *avg_storage;
    avg_storage = (double *) malloc(params.peers * n_dims * sizeof(double));
    fill_n(avg_storage, params.peers * n_dims, 0);
    avgsummaries = (double **) malloc(params.peers * sizeof(double *));

    for (int i = 0; i < params.peers; ++i) {
        avgsummaries[i] = &avg_storage[i * n_dims];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        if (peerID == 0) {
            int pts_count = peerLastItem[0] + 1;
            double weight = 1 / (double) pts_count;
            for (int i = 0; i < peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    avgsummaries[peerID][j] += weight * data[i][j];
                }
            }
        } else {
            int pts_count = peerLastItem[peerID] - peerLastItem[peerID-1];
            double weight = 1 / (double) pts_count;
            for (int i = peerLastItem[peerID-1]; i < peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    avgsummaries[peerID][j] += weight * data[i][j];
                }
            }
            if (peerID == params.peers-1) {
                for (int j = 0; j < n_dims; ++j) {
                    avgsummaries[peerID][j] += weight * data[peerLastItem[peerID]][j];
                }
            }
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
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize-1);
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
/*
    for(int peerID = 0; peerID < params.peers; peerID++){
        cout << peerID << ": ";
        for (int i = 0; i < n_dims; ++i) {
            cout << avgsummaries[peerID][i] << ", ";
        }
        cout << endl;
    }
*/
    for(int peerID = 0; peerID < params.peers; peerID++){
        if (peerID == 0) {
            for (int i = 0; i < peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    data[i][j] -= avgsummaries[peerID][j];
                }
            }
        } else {
            for (int i = peerLastItem[peerID-1]; i < peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    data[i][j] -= avgsummaries[peerID][j];
                }
            }
            if (peerID == params.peers-1) {
                for (int j = 0; j < n_dims; ++j) {
                    data[peerLastItem[peerID]][j] -= avgsummaries[peerID][j];
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

    auto pcc_start = chrono::steady_clock::now();

    double *pcc_storage, **pcc_i, ***pcc;
    pcc_storage = (double *) malloc(params.peers * n_dims * n_dims * sizeof(double));
    if (pcc_storage == nullptr) {
        cout << "Malloc error on pcc_storage" << endl;
        exit(-1);
    }
    fill_n(pcc_storage, params.peers * n_dims * n_dims, 0);
    pcc_i = (double **) malloc(params.peers * n_dims * sizeof(double *));
    if (pcc_i == nullptr) {
        cout << "Malloc error on pcc_i" << endl;
        exit(-1);
    }
    pcc = (double ***) malloc(params.peers * sizeof(double **));
    if (pcc == nullptr) {
        cout << "Malloc error on pcc" << endl;
        exit(-1);
    }

    for (int i = 0; i < params.peers * n_dims; ++i) {
        pcc_i[i] = &pcc_storage[i * n_dims];
    }
    for (int i = 0; i < params.peers; ++i) {
        pcc[i] = &pcc_i[i * n_dims];
    }

    double **squaresum_dims, *squaresum_dims_storage;
    squaresum_dims_storage = (double *) malloc(params.peers * n_dims * sizeof(double));
    fill_n(squaresum_dims_storage, params.peers * n_dims, 0);
    squaresum_dims = (double **) malloc(params.peers * sizeof(double *));

    for (int i = 0; i < params.peers; ++i) {
        squaresum_dims[i] = &squaresum_dims_storage[i * n_dims];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        for (int l = 0; l < n_dims; ++l) {
            pcc[peerID][l][l] = 1;
            if (peerID == 0) {
                for (int i = 0; i < peerLastItem[peerID]; ++i) {
                    squaresum_dims[peerID][l] += pow(data[i][l], 2);
                }
            } else {
                for (int i = peerLastItem[peerID-1]; i < peerLastItem[peerID]; ++i) {
                    squaresum_dims[peerID][l] += pow(data[i][l], 2);
                }
                if (peerID == params.peers-1) {
                    squaresum_dims[peerID][l] += pow(data[peerLastItem[peerID]][l], 2);
                }
            }
            for (int m = l + 1; m < n_dims; ++m) {
                if (peerID == 0) {
                    for (int i = 0; i < peerLastItem[peerID]; ++i) {
                        pcc[peerID][l][m] += data[i][l] * data[i][m];
                    }
                } else {
                    for (int i = peerLastItem[peerID-1]; i < peerLastItem[peerID]; ++i) {
                        pcc[peerID][l][m] += data[i][l] * data[i][m];
                    }
                    if (peerID == params.peers-1) {
                        pcc[peerID][l][m] += data[peerLastItem[peerID]][l] * data[peerLastItem[peerID]][m];
                    }
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
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize-1);
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

    auto partition_start = chrono::steady_clock::now();

    int *uncorr_vars, *corr_vars;
    corr_vars = (int *) malloc(params.peers * sizeof(int));
    uncorr_vars = (int *) malloc(params.peers * sizeof(int));

    for(int peerID = 0; peerID < params.peers; peerID++) {
        corr_vars[peerID] = 0;
        uncorr_vars[peerID] = 0;
        for (int i = 0; i < n_dims; ++i) {
            double overall = 0.0;
            for (int j = 0; j < n_dims; ++j) {
                if (i != j) {
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

    int *uncorr_storage, **uncorr;
    uncorr_storage = (int *) malloc(params.peers * uncorr_vars[0] * sizeof(int));
    if (pcc_storage == nullptr) {
        cout << "Malloc error on uncorr_storage" << endl;
        exit(-1);
    }
    uncorr = (int **) (malloc(params.peers * sizeof(int *)));
    if (uncorr == nullptr) {
        cout << "Malloc error on corr or uncorr" << endl;
        exit(-1);
    }
    for (int i = 0; i < params.peers; ++i) {
        uncorr[i] = &uncorr_storage[i * uncorr_vars[0]];
    }
    double ***corr;
    corr = (double ***) malloc(params.peers * sizeof(double **));
    if (corr == nullptr) {
        cout << "Malloc error on corr" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        int npts = 0;
        if (peerID == 0) {
            npts = peerLastItem[peerID];
        } else {
            npts = peerLastItem[peerID] - peerLastItem[peerID-1];
            if (peerID == params.peers-1) {
                npts++;
            }
        }
        corr[peerID] = (double **) malloc(corr_vars[peerID] * sizeof(double *));
        if (corr[peerID] == nullptr) {
            cout << "Malloc error on corr for peer " << peerID << endl;
            exit(-1);
        }
        for (int k = 0; k < corr_vars[peerID]; ++k) {
            corr[peerID][k] = (double *) malloc(npts * sizeof(double));
            if (corr[peerID][k] == nullptr) {
                cout << "Malloc error on corr for peer " << peerID << endl;
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
                    for (int k = 0; k < peerLastItem[peerID]; ++k) {
                        corr[peerID][corr_vars[peerID]][elem] = data[k][i];
                        elem++;
                    }
                } else {
                    for (int k = peerLastItem[peerID-1]; k < peerLastItem[peerID]; ++k) {
                        corr[peerID][corr_vars[peerID]][elem] = data[k][i];
                        elem++;
                    }
                    if (peerID == params.peers-1) {
                        corr[peerID][corr_vars[peerID]][elem] = data[peerLastItem[peerID]][i];
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

    auto pca_start = chrono::steady_clock::now();

    //Define the structure to save PC1_corr and PC2_corr
    double *newspace_storage, **newspace_i, ***newspace;
    newspace_storage = (double *) malloc(params.peers * corr_vars[0] * corr_vars[0] * sizeof(double));
    if (newspace_storage == nullptr) {
        cout << "Malloc error on newspace_storage" << endl;
        exit(-1);
    }
    newspace_i = (double **) malloc(params.peers * corr_vars[0] * sizeof(double *));
    if (newspace_i == nullptr) {
        cout << "Malloc error on newspace_i" << endl;
        exit(-1);
    }
    newspace = (double ***) malloc(params.peers * sizeof(double **));
    if (newspace == nullptr) {
        cout << "Malloc error on newspace" << endl;
        exit(-1);
    }

    for (int i = 0; i < params.peers * corr_vars[0]; ++i) {
        newspace_i[i] = &newspace_storage[i * corr_vars[0]];
    }
    for (int i = 0; i < params.peers; ++i) {
        newspace[i] = &newspace_i[i * corr_vars[0]];
    }

    for(int peerID = 0; peerID < params.peers; peerID++){
        int npts = 0;
        if (peerID == 0) {
            npts = peerLastItem[peerID];
        } else {
            npts = peerLastItem[peerID] - peerLastItem[peerID-1];
            if (peerID == params.peers-1) {
                npts++;
            }
        }
        //PCA_transform(corr[peerID], corr_vars[peerID], npts, newspace[peerID]);
        for (int i=0; i < corr_vars[peerID]; i++) {
            for (int j=0;j<corr_vars[peerID];j++) {
                newspace[peerID][i][j]=0;
                for (int k=0;k<npts;k++) {
                    newspace[peerID][i][j] += corr[peerID][i][k] * corr[peerID][j][k];
                }
                newspace[peerID][i][j] = newspace[peerID][i][j] / (npts - 1);
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
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize-1);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                for (int l = 0; l < corr_vars[peerID]; ++l) {
                    for (int k = 0; k < corr_vars[peerID]; ++k) {
                        newspace[peerID][l][k] = (newspace[peerID][l][k] + newspace[neighborID][l][k]) / 2;
                    }
                }
                memcpy(newspace[neighborID][0], newspace[peerID][0], corr_vars[peerID] * corr_vars[peerID] * sizeof(double));
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
        mat x(corr_vars[peerID], corr_vars[peerID]);
        for (int i=0; i < corr_vars[peerID]; i++) {
            for (int j=0;j<corr_vars[peerID];j++) {
                x(i,j) = newspace[peerID][i][j];
            }
        }
        cx_vec eigval;
        cx_mat eigvec;

        eig_gen(eigval, eigvec, x, "balance");
        cout << peerID << endl;
        cout << "eigval" << endl;
        eigval.print();
        cout << "eigvect" << endl;
        eigvec.print();
        //ei_gen results are ordered by magnitude. take the first and second
    }

    auto pca_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to compute Principal Component Analysis on CORR set: " <<
             chrono::duration_cast<chrono::nanoseconds>(pca_end - pca_start).count()*1e-9 << endl;
    }

    /*
    auto _start = chrono::steady_clock::now();



    auto _end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to compute Pearson matrix: " <<
             chrono::duration_cast<chrono::nanoseconds>(_end - _start).count()*1e-9 << endl;
    }

    auto _start = chrono::steady_clock::now();



    auto _end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to compute Pearson matrix: " <<
             chrono::duration_cast<chrono::nanoseconds>(_end - _start).count()*1e-9 << endl;
    }
*/
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
        << "-ethr       threshold for the selection of optimal number of clusters in Elbow method\n"
        << "-incircle   percentage of points in a cluster to be considered as inliers\n"
        << "-subspace   percentage of subspaces in which a point must be to consider as outliers\n"
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