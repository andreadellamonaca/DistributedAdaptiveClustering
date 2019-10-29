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

cluster_report TT;
void data_out(double ****data, long *lastitem, string name, bool ***incircle, int peers, int cs, cluster_report *report) {
    fstream fout;
    fout.open("../" + name, ios::out | ios::trunc);

    for (int k = 0; k < cs; ++k) {
        for (int peerid = 0; peerid < peers; ++peerid) {
            int pts_count = 0;
            if (peerid == 0) {
                pts_count = lastitem[0] + 1;
            } else {
                pts_count = lastitem[peerid] - lastitem[peerid-1];
            }
            for (int i = 0; i < pts_count; ++i) {
                for (int j = 0; j < 2; ++j) {
                    fout << data[peerid][k][j][i] << ",";
                }
                if (incircle[peerid][k][i]) {
                    fout << "1,";
                } else {
                    fout << "0,";
                }
                fout << report[peerid].cidx[i];
                fout << "\n";
            }
        }
    }
    fout.close();
    fstream fout2;
    fout2.open("../centroids_" + name, ios::out | ios::trunc);
    for (int i = 0; i < report[0].k; ++i) {
        fout2 << report[0].centroids.at(0, i) << ",";
        fout2 << report[0].centroids.at(1, i) << "\n";
    }
    fout2.close();
}

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
            for (int i = 0; i <= peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    avgsummaries[peerID][j] += weight * data[i][j];
                }
            }
        } else {
            int pts_count = peerLastItem[peerID] - peerLastItem[peerID-1];
            double weight = 1 / (double) pts_count;
            for (int i = peerLastItem[peerID-1] + 1; i <= peerLastItem[peerID]; ++i) {
                for (int j = 0; j < n_dims; ++j) {
                    avgsummaries[peerID][j] += weight * data[i][j];
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
                for (int i = 0; i <= peerLastItem[peerID]; ++i) {
                    squaresum_dims[peerID][l] += pow(data[i][l], 2);
                }
            } else {
                for (int i = peerLastItem[peerID-1] + 1; i <= peerLastItem[peerID]; ++i) {
                    squaresum_dims[peerID][l] += pow(data[i][l], 2);
                }
            }
            for (int m = l + 1; m < n_dims; ++m) {
                if (peerID == 0) {
                    for (int i = 0; i <= peerLastItem[peerID]; ++i) {
                        pcc[peerID][l][m] += data[i][l] * data[i][m];
                    }
                } else {
                    for (int i = peerLastItem[peerID-1] + 1; i <= peerLastItem[peerID]; ++i) {
                        pcc[peerID][l][m] += data[i][l] * data[i][m];
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
            npts = peerLastItem[peerID] + 1;
        } else {
            npts = peerLastItem[peerID] - peerLastItem[peerID-1];
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
            npts = peerLastItem[peerID] + 1;
        } else {
            npts = peerLastItem[peerID] - peerLastItem[peerID-1];
        }

        for (int i=0; i < corr_vars[peerID]; i++) {
            for (int j=i;j<corr_vars[peerID];j++) {
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
                igraph_vector_remove_section(&neighbors, params.fanOut, neighborsSize);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];
                igraph_integer_t edgeID;
                igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                for (int l = 0; l < corr_vars[peerID]; ++l) {
                    for (int k = l; k < corr_vars[peerID]; ++k) {
                        newspace[peerID][l][k] = (newspace[peerID][l][k] + newspace[neighborID][l][k]) / 2;
                        newspace[peerID][k][l] = newspace[peerID][l][k];
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

    double ***combine;
    combine = (double ***) malloc(params.peers * sizeof(double **));
    if (combine == nullptr) {
        cout << "Malloc error on combine" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        int npts = 0;
        if (peerID == 0) {
            npts = peerLastItem[peerID] + 1;
        } else {
            npts = peerLastItem[peerID] - peerLastItem[peerID-1];
        }

        combine[peerID] = (double **) malloc(3 * sizeof(double *));
        if (combine[peerID] == nullptr) {
            cout << "Malloc error on combine for peer " << peerID << endl;
            exit(-1);
        }
        for (int k = 0; k < 3; ++k) {
            combine[peerID][k] = (double *) malloc(npts * sizeof(double));
            if (combine[peerID][k] == nullptr) {
                cout << "Malloc error on combine for peer " << peerID << endl;
                exit(-1);
            }
        }

        mat cov_mat(newspace[peerID][0], corr_vars[peerID], corr_vars[peerID]);
        vec eigval;
        mat eigvec;
        eig_sym(eigval, eigvec, cov_mat);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < npts; ++j) {
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
    free(newspace_storage);
    free(newspace_i);
    free(newspace);

    auto pca_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to compute Principal Component Analysis on CORR set: " <<
             chrono::duration_cast<chrono::nanoseconds>(pca_end - pca_start).count()*1e-9 << endl;
    }

    auto cs_start = chrono::steady_clock::now();

    double ****subspace;
    subspace = (double ****) malloc(params.peers * sizeof(double ***));
    if (subspace == nullptr) {
        cout << "Malloc error on subspace" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        subspace[peerID] = (double ***) malloc(uncorr_vars[peerID] * sizeof(double **));
        int npts = 0;
        if (peerID == 0) {
            npts = peerLastItem[peerID] + 1;
        } else {
            npts = peerLastItem[peerID] - peerLastItem[peerID-1];
        }
        for (int m = 0; m < uncorr_vars[peerID]; ++m) {
            subspace[peerID][m] = (double **) malloc(2 * sizeof(double *));
            for (int k = 0; k < 2; ++k) {
                subspace[peerID][m][k] = (double *) malloc(npts * sizeof(double ));
            }
        }
    }

    for (int m = 0; m < uncorr_vars[0]; ++m) {
        double *covar_storage, **covar_i, ***covar;
        covar_storage = (double *) malloc(params.peers * 3 * 3 * sizeof(double));
        if (covar_storage == nullptr) {
            cout << "Malloc error on covar_storage" << endl;
            exit(-1);
        }
        covar_i = (double **) malloc(params.peers * 3 * sizeof(double *));
        if (covar_i == nullptr) {
            cout << "Malloc error on covar_i" << endl;
            exit(-1);
        }
        covar = (double ***) malloc(params.peers * sizeof(double **));
        if (covar == nullptr) {
            cout << "Malloc error on covar" << endl;
            exit(-1);
        }

        for (int j = 0; j < params.peers * 3; ++j) {
            covar_i[j] = &covar_storage[j * 3];
        }
        for (int j = 0; j < params.peers; ++j) {
            covar[j] = &covar_i[j * 3];
        }

        for(int peerID = 0; peerID < params.peers; peerID++) {
            int npts = 0;
            if (peerID == 0) {
                npts = peerLastItem[peerID] + 1;
            } else {
                npts = peerLastItem[peerID] - peerLastItem[peerID-1];
            }
            for (int j = 0; j < npts; ++j) {
                if (peerID == 0) {
                    combine[peerID][2][j] = data[j][uncorr[peerID][m]];
                } else {
                    int index = peerLastItem[peerID-1] + 1 + j;
                    combine[peerID][2][j] = data[index][uncorr[peerID][m]];
                }
            }

            for (int l=0; l < 3; l++) {
                for (int j=l; j < 3;j++) {
                    covar[peerID][l][j]=0;
                    for (int k=0;k<npts;k++) {
                        covar[peerID][l][j] += combine[peerID][l][k] * combine[peerID][j][k];
                    }
                    covar[peerID][l][j] = covar[peerID][l][j] / (npts - 1);
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
            mat cov_mat(covar[peerID][0], 3, 3);
            vec eigval;
            mat eigvec;
            eig_sym(eigval, eigvec, cov_mat);

            int npts = 0;
            if (peerID == 0) {
                npts = peerLastItem[peerID] + 1;
            } else {
                npts = peerLastItem[peerID] - peerLastItem[peerID-1];
            }

            for (int j = 0; j < npts; ++j) {
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
    }
    free(combine);

    auto cs_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to create candidate subspaces: " <<
             chrono::duration_cast<chrono::nanoseconds>(cs_end - cs_start).count()*1e-9 << endl;
    }

    auto clustering_start = chrono::steady_clock::now();

    bool ***incircle;
    incircle = (bool ***) malloc(params.peers * sizeof(bool **));
    if (incircle == nullptr) {
        cout << "Malloc error on incircle" << endl;
        exit(-1);
    }

    for(int peerID = 0; peerID < params.peers; peerID++) {
        incircle[peerID] = (bool **) malloc(uncorr_vars[peerID] * sizeof(bool *));
        int npts = 0;
        if (peerID == 0) {
            npts = peerLastItem[peerID] + 1;
        } else {
            npts = peerLastItem[peerID] - peerLastItem[peerID-1];
        }
        for (int m = 0; m < uncorr_vars[peerID]; ++m) {
            incircle[peerID][m] = (bool *) malloc(npts * sizeof(bool));
            fill_n(incircle[peerID][m], npts, false);
        }
    }

    cluster_report *final_i = (cluster_report *) calloc(uncorr_vars[0] * params.peers, sizeof(cluster_report));
    cluster_report **final = (cluster_report **) calloc(uncorr_vars[0], sizeof(cluster_report*));
    for (int m = 0; m < uncorr_vars[0]; ++m) {
        final[m] = &final_i[m * params.peers];
    }

    for (int i = 0; i < uncorr_vars[0]; ++i) {
        cluster_report *prev = (cluster_report *) calloc(params.peers, sizeof(cluster_report));
        for (int j = 1; j <= K_MAX; ++j) {
            mat w_k(params.peers, j);
            w_k.fill(0);
            cube actual_centroids(2, j, params.peers);
            cube prev_centroids(2, j, params.peers);
            double *prev_err = (double *) calloc(params.peers, sizeof(double));
            double *error = (double *) calloc(params.peers, sizeof(double));
            fill_n(error, params.peers, 1e9);

            // Reset parameters for convergence estimate
            Numberofconverged = params.peers;
            fill_n(converged, params.peers, false);
            fill_n(convRounds, params.peers, 0);
            rounds = 0;

            while( (params.roundsToExecute < 0 && Numberofconverged) || params.roundsToExecute > 0){
                prev_centroids = actual_centroids;
                memcpy(prev_err, error, params.peers * sizeof(double));
                fill_n(error, params.peers, 0);
                for(int peerID = 0; peerID < params.peers; peerID++){
                    // check peer convergence
                    if(params.roundsToExecute < 0 && converged[peerID])
                        continue;

                    int npts = 0;
                    if (peerID == 0) {
                        npts = peerLastItem[peerID] + 1;
                    } else {
                        npts = peerLastItem[peerID] - peerLastItem[peerID - 1];
                    }
                    int *clusterid = (int *) calloc(npts, sizeof(int));
                    for (int k = 0; k < npts; ++k) {
                        clusterid[k] = 0;
                        double mindist = L2distance(actual_centroids.at(0, 0, peerID), actual_centroids.at(1, 0, peerID), subspace[peerID][i][0][k], subspace[peerID][i][1][k]);
                        for (int l = 1; l < j; ++l) {
                            double dist = L2distance(actual_centroids.at(0, l, peerID), actual_centroids.at(1, l, peerID), subspace[peerID][i][0][k], subspace[peerID][i][1][k]);
                            if ( dist < mindist) {
                                clusterid[k] = l;
                            }
                        }
                        w_k.at(peerID, clusterid[k]) += 1;
                        actual_centroids.at(0, clusterid[k], peerID) += subspace[peerID][i][0][k];
                        actual_centroids.at(1, clusterid[k], peerID) += subspace[peerID][i][1][k];
                    }
                    for (int k = 0; k < npts; ++k) {
                        error[peerID] += L2distance(actual_centroids.at(0, clusterid[k], peerID) / w_k.at(peerID, clusterid[k]), actual_centroids.at(1, clusterid[k] / w_k.at(peerID, clusterid[k]), peerID), subspace[peerID][i][0][k], subspace[peerID][i][1][k]);
                    }
                    free(clusterid);
                }

                // Reset parameters for convergence estimate
                fill_n(dimestimate, params.peers, 0);
                dimestimate[0] = 1;
                int N_converged = params.peers;
                bool *converged2 = (bool *) calloc(params.peers, sizeof(bool));
                fill_n(converged2, params.peers, false);
                int *convRounds2 = (int *) calloc(params.peers, sizeof(int));
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
                        for(int k = 0; k < neighborsSize; k++){
                            int neighborID = (int) VECTOR(neighbors)[k];
                            igraph_integer_t edgeID;
                            igraph_get_eid(&graph, &edgeID, peerID, neighborID, IGRAPH_UNDIRECTED, 1);

                            for (int l = 0; l < j; ++l) {
                                actual_centroids.at(0, l, peerID) = (actual_centroids.at(0, l, peerID) + actual_centroids.at(0, l, neighborID)) / 2;
                                actual_centroids.at(1, l, peerID) = (actual_centroids.at(1, l, peerID) + actual_centroids.at(1, l, neighborID)) / 2;
                                w_k.at(peerID, l) = (w_k.at(peerID, l) + w_k.at(neighborID, l)) / 2;
                            }
                            error[peerID] = (error[peerID] + error[neighborID]) / 2;

                            actual_centroids.slice(neighborID) = actual_centroids.slice(peerID);
                            w_k.at(neighborID) = w_k.at(peerID);
                            memcpy(&error[neighborID], &error[peerID], sizeof(double));
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

                for(int peerID = 0; peerID < params.peers; peerID++) {
                    for (int k = 0; k < j; ++k) {
                        actual_centroids.at(0, k, peerID) = actual_centroids.at(0, k, peerID) / w_k.at(peerID, k);
                        actual_centroids.at(1, k, peerID) = actual_centroids.at(1, k, peerID) / w_k.at(peerID, k);
                    }
                }

                // check local convergence
                if (params.roundsToExecute < 0) {
                    for(int peerID = 0; peerID < params.peers; peerID++){
                        if(converged[peerID])
                            continue;

                        converged[peerID] = ((prev_err[peerID] - error[peerID]) / prev_err[peerID] <= 0.1);

                        if(converged[peerID]){
                            Numberofconverged --;
                        }
                    }
                }
                rounds++;
                //cerr << "\r Active peers: " << Numberofconverged << " - Rounds: " << rounds << "          ";
                params.roundsToExecute--;
            }
            free(prev_err);
            free(error);
//            cout << j << endl;
//            for(int peerID = 0; peerID < params.peers; peerID++){
//                cout << peerID << ": ";
//                for (int k = 0; k < j; ++k) {
//                    cout << actual_centroids.at(0, k, peerID) << ", " << actual_centroids.at(1, k, peerID) << " | ";
//                }
//                cout << endl;
//            }

            for(int peerID = 0; peerID < params.peers; peerID++){
                int npts = 0;
                if (peerID == 0) {
                    npts = peerLastItem[peerID] + 1;
                } else {
                    npts = peerLastItem[peerID] - peerLastItem[peerID - 1];
                }
                if (j == 1) {
                    prev[peerID].cidx = (int *) malloc(npts * sizeof(int));
                    final[i][peerID].cidx = (int *) malloc(npts * sizeof(int));
                    final[i][peerID].centroids = actual_centroids.slice(peerID);
                    final[i][peerID].k = j;
                    final[i][peerID].BetaCV = 0.0;
                    fill_n(final[i][peerID].cidx, npts, 0);
                } else {
                    final[i][peerID].centroids = actual_centroids.slice(peerID);
                    final[i][peerID].k = j;
                    create_cidx_matrix(subspace[peerID][i], npts, final[i][peerID]);
                }
            }

            if (j > 1) {
                double *pts_storage = (double *) calloc(params.peers * j, sizeof(double));
                fill_n(pts_storage, params.peers * j, 0);
                double **pts_incluster = (double **) calloc(params.peers, sizeof(double*));
                double *c_mean_storage = (double *) calloc(params.peers * 2, sizeof(double));
                double **c_mean = (double **) calloc(params.peers, sizeof(double*));
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    pts_incluster[peerID] = &pts_storage[peerID * j];
                    c_mean[peerID] = &c_mean_storage[peerID * 2];
                    int pts_count = 0;
                    if (peerID == 0) {
                        pts_count = peerLastItem[peerID] + 1;
                    } else {
                        pts_count = peerLastItem[peerID] - peerLastItem[peerID-1];
                    }
                    double weight = 1 / (double) pts_count;
                    for (int k = 0; k < pts_count; ++k) {
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

                double *bcss_storage = (double *) calloc(params.peers, sizeof(double));
                double *wcss_storage = (double *) calloc(params.peers, sizeof(double));
                double *nin_storage = (double *) calloc(params.peers, sizeof(double));
                double *nout_storage = (double *) calloc(params.peers, sizeof(double));
                fill_n(nin_storage, params.peers, 0);
                fill_n(nout_storage, params.peers, 0);
                fill_n(bcss_storage, params.peers, 0);
                fill_n(wcss_storage, params.peers, 0);
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    int pts_count = 0;
                    if (peerID == 0) {
                        pts_count = peerLastItem[peerID] + 1;
                    } else {
                        pts_count = peerLastItem[peerID] - peerLastItem[peerID-1];
                    }
                    for (int m = 0; m < j; ++m) {
                        nin_storage[peerID] += pts_incluster[peerID][m] * (pts_incluster[peerID][m] - 1);
                        for (int k = 0; k < j; ++k) {
                            if (k != m) {
                                nout_storage[peerID] += pts_incluster[peerID][m] * pts_incluster[peerID][k];
                            }
                        }
                        bcss_storage[peerID] += (pts_incluster[peerID][m] * L2distance(final[i][peerID].centroids.at(0, m), final[i][peerID].centroids.at(1, m), c_mean[peerID][0], c_mean[peerID][1]));
                        for (int k = 0; k < pts_count; ++k) {
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
                    double metric = (nout_storage[peerID] * wcss_storage[peerID]) / (nin_storage[peerID] * bcss_storage[peerID]);
                    final[i][peerID].BetaCV = metric;
                }
                free(nout_storage);
                free(wcss_storage);
                free(nin_storage);
                free(bcss_storage);
                double val = fabs(prev[0].BetaCV - final[i][0].BetaCV);
                if (/*fabs(prev[0].BetaCV - final[i][0].BetaCV) <= ELBOW_THRES*/ final[i][0].k == 3) {
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

        double *inliers = (double *) calloc(params.peers, sizeof(double));
        double *prev_inliers = (double *) calloc(params.peers, sizeof(double));
        double *cluster_dim = (double *) calloc(params.peers, sizeof(double));

        for (int l = 0; l < final[i][0].k; ++l) {
            for (int peerID = 0; peerID < params.peers; peerID++) {
                int pts_count = 0;
                if (peerID == 0) {
                    pts_count = peerLastItem[peerID] + 1;
                } else {
                    pts_count = peerLastItem[peerID] - peerLastItem[peerID-1];
                }
                cluster_dim[peerID] = cluster_size(final[i][peerID], l, pts_count);
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
                        cluster_dim[peerID] = (cluster_dim[peerID] + cluster_dim[neighborID]) / 2;
                        memcpy(&cluster_dim[neighborID], &cluster_dim[peerID], sizeof(double));
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

            int N_converged = params.peers;
            fill_n(inliers, params.peers, 0);
            bool *converged2 = (bool *) calloc(params.peers, sizeof(bool));
            fill_n(converged2, params.peers, false);
            double *actual_dist = (double *) calloc(params.peers, sizeof(double));
            double *actual_cluster_dim = (double *) calloc(params.peers, sizeof(double));

            while (N_converged) {
                memcpy(prev_inliers, inliers, params.peers * sizeof(double));
                fill_n(actual_dist, params.peers, 0.0);
                fill_n(actual_cluster_dim, params.peers, 0.0);
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    if (converged2[peerID])
                        continue;
                    int pts_count = 0;
                    if (peerID == 0) {
                        pts_count = peerLastItem[peerID] + 1;
                    } else {
                        pts_count = peerLastItem[peerID] - peerLastItem[peerID-1];
                    }
                    for (int k = 0; k < pts_count; ++k) {
                        if (final[i][peerID].cidx[k] == l && !(incircle[peerID][i][k]) ) {
                            actual_dist[peerID] += L2distance(final[i][peerID].centroids.at(0, l), final[i][peerID].centroids.at(1, l), subspace[peerID][i][0][k], subspace[peerID][i][1][k]);
                            actual_cluster_dim[peerID]++;
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

                            actual_dist[peerID] = (actual_dist[peerID] + actual_dist[neighborID]) / 2;
                            actual_cluster_dim[peerID] = (actual_cluster_dim[peerID] + actual_cluster_dim[neighborID]) / 2;

                            memcpy(&actual_dist[neighborID], &actual_dist[peerID], sizeof(double));
                            memcpy(&actual_cluster_dim[neighborID], &actual_cluster_dim[peerID], sizeof(double));
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
                    double dist_mean = actual_dist[peerID] / actual_cluster_dim[peerID];
                    int pts_count = 0;
                    if (peerID == 0) {
                        pts_count = peerLastItem[peerID] + 1;
                    } else {
                        pts_count = peerLastItem[peerID] - peerLastItem[peerID-1];
                    }
                    for (int k = 0; k < pts_count; ++k) {
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

                            inliers[peerID] = (inliers[peerID] + inliers[neighborID]) / 2;
                            memcpy(&inliers[neighborID], &inliers[peerID], sizeof(double));

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

                // check local convergence
                for (int peerID = 0; peerID < params.peers; peerID++) {
                    if (converged2[peerID])
                        continue;

                    converged2[peerID] = ( (inliers[peerID] >= PERCENTAGE_INCIRCLE * cluster_dim[peerID])
                            || prev_inliers[peerID] == inliers[peerID] );
                    if (converged2[peerID]) {
                        N_converged--;
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
        int pts_count = 0;
        if (peerID == 0) {
            pts_count = peerLastItem[peerID] + 1;
        } else {
            pts_count = peerLastItem[peerID] - peerLastItem[peerID-1];
        }
        for (int k = 0; k < pts_count; ++k) {
            int occurrence = 0;
            for (int j = 0; j < uncorr_vars[peerID]; ++j) {
                if (!incircle[peerID][j][k]) {
                    occurrence++;
                }
            }

            if (occurrence >= std::round(uncorr_vars[peerID] * PERCENTAGE_SUBSPACES)) {
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
    free(uncorr_vars);
    free(incircle);

    auto clustering_end = chrono::steady_clock::now();
    if (!outputOnFile) {
        cout << "Time (s) required to run K-Means and identify outliers: " <<
             chrono::duration_cast<chrono::nanoseconds>(clustering_end - clustering_start).count()*1e-9 << endl;
    }

//    auto _start = chrono::steady_clock::now();
//
//    auto _end = chrono::steady_clock::now();
//    if (!outputOnFile) {
//        cout << "Time (s) required to compute Pearson matrix: " <<
//             chrono::duration_cast<chrono::nanoseconds>(_end - _start).count()*1e-9 << endl;
//    }

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