#include "adaptive_clustering.h"
/*
int main() {
    cout << fixed;
    cout << setprecision(5);

    int N_DIMS; //Number of dimensions
    int N_DATA; //Number of observations

    //Define the structure to load the dataset
    double **data, *data_storage;
    getDatasetDims(filename, &N_DIMS, &N_DATA);

    data_storage = (double *) malloc(N_DIMS * N_DATA * sizeof(double));
    if (data_storage == nullptr) {
        cout << "Malloc error on data_storage" << endl;
        exit(-1);
    }
    data = (double **) malloc(N_DIMS * sizeof(double *));
    if (data == nullptr) {
        cout << "Malloc error on data" << endl;
        exit(-1);
    }

    for (int i = 0; i < N_DIMS; ++i) {
        data[i] = &data_storage[i * N_DATA];
    }

    // Fill the structure from csv
    loadData(filename, data, N_DIMS);

    cout << "Dataset Loaded" << endl;
//    for (int i = 0; i < N_DIMS; ++i) {
//        for(int j = 0; j < N_DATA; j++) {
//            cout << data[i][j] << " ";
//        }
//       cout << "\n";
//    }

    auto start = chrono::steady_clock::now();

    //Standardization
    Standardize_dataset(data, N_DIMS, N_DATA);

    cout << "Dataset standardized" << endl;

    //Define the structure to load the Correlation Matrix
    double **pearson, *pearson_storage;
    pearson_storage = (double *) malloc(N_DIMS * N_DIMS * sizeof(double));
    if (pearson_storage == nullptr) {
        cout << "Malloc error on pearson_storage" << endl;
        exit(-1);
    }
    pearson = (double **) malloc(N_DIMS * sizeof(double *));
    if (pearson == nullptr) {
        cout << "Malloc error on pearson" << endl;
        exit(-1);
    }

    for (int i = 0; i < N_DIMS; ++i) {
        pearson[i] = &pearson_storage[i * N_DIMS];
    }

    for (int i = 0; i < N_DIMS; ++i) {
        pearson[i][i] = 1;
        for (int j = i+1; j < N_DIMS; ++j) {
            double value = PearsonCoefficient(data[i], data[j], N_DATA);
            pearson[i][j] = value;
            pearson[j][i] = value;
        }
    }

    cout << "Pearson Correlation Coefficient computed" << endl;
//    for (int i = 0; i < N_DIMS; ++i) {
//        for(int j = 0; j < N_DIMS; j++) {
//            cout << pearson[i][j] << " ";
//        }
//        cout << "\n";
//    }

    //Dimensions partitioned in CORR and UNCORR, then CORR U UNCORR = DIMS
    int corr_vars = 0, uncorr_vars = 0;
    for (int i = 0; i < N_DIMS; ++i) {
        double overall = 0.0;
        for (int j = 0; j < N_DIMS; ++j) {
            if (i != j) {
                overall += pearson[i][j];
            }
        }
        if ((overall/N_DIMS) >= 0) {
            corr_vars++;
        } else {
            uncorr_vars++;
        }
    }

    if (corr_vars < 2) {
        cout << "Error: Correlated dimensions must be more than 1 in order to apply PCA!" << endl;
        exit(-1);
    }
    if (uncorr_vars == 0) {
        cout << "Error: There are no candidate subspaces!" << endl;
        exit(-1);
    }

    double **corr;
    int *uncorr;
    corr = (double **) malloc(corr_vars * sizeof(double *));
    if (corr == nullptr) {
        cout << "Malloc error on corr" << endl;
        exit(-1);
    }
    uncorr = (int *) (malloc(uncorr_vars * sizeof(int)));
    if (uncorr == nullptr) {
        cout << "Malloc error on corr or uncorr" << endl;
        exit(-1);
    }

    corr_vars = 0, uncorr_vars = 0;
    for (int i = 0; i < N_DIMS; ++i) {
        double overall = 0.0;
        for (int j = 0; j < N_DIMS; ++j) {
            if (i != j) {
                overall += pearson[i][j];
            }
        }
        if ((overall/N_DIMS) >= 0) {
            corr[corr_vars] = data[i];
            corr_vars++;
        } else {
            uncorr[uncorr_vars] = i;
            uncorr_vars++;
        }
    }

    free(pearson_storage);
    free(pearson);
    cout << "Correlated dimensions: " << corr_vars << ", " << "Uncorrelated dimensions: " << uncorr_vars << endl;

//    cout << "CORR subspace" << endl;
//    for (int i = 0; i < corr_vars; ++i) {
//        for(int j = 0; j < N_DATA; j++) {
//            cout << corr[i][j] << " ";
//        }
//        cout << "\n";
//    }
//
//    cout << "UNCORR subspace" << endl;
//    for (int i = 0; i < uncorr_vars; ++i) {
//        cout << uncorr[i] << " ";
//    }
//    cout << "\n";

    //Define the structure to save PC1_corr and PC2_corr ------ matrix [2 * N_DATA]
    double **newspace, *newspace_storage;
    newspace_storage = (double *) malloc(N_DATA * 2 * sizeof(double));
    if (newspace_storage == nullptr) {
        cout << "Malloc error on newspace_storage" << endl;
        exit(-1);
    }
    newspace = (double **) malloc(2 * sizeof(double *));
    if (newspace == nullptr) {
        cout << "Malloc error on newspace" << endl;
        exit(-1);
    }
    for (int i = 0; i < 2; ++i) {
        newspace[i] = &newspace_storage[i*N_DATA];
    }

    PCA_transform(corr, corr_vars, N_DATA, newspace);

    free(corr);
    cout << "PCA computed on CORR subspace" << endl;

    //Define the structure to save the candidate subspaces ------ #UNCORR * matrix [2 * N_DATA]
    double *cs_storage, **csi, ***cs;
    cs_storage = (double *) malloc(N_DATA * 2 * uncorr_vars * sizeof(double));
    if (cs_storage == nullptr) {
        cout << "Malloc error on cs_storage" << endl;
        exit(-1);
    }
    csi = (double **) malloc(2 * uncorr_vars * sizeof(double *));
    if (csi == nullptr) {
        cout << "Malloc error on csi" << endl;
        exit(-1);
    }
    cs = (double ***) malloc(uncorr_vars * sizeof(double **));
    if (cs == nullptr) {
        cout << "Malloc error on cs" << endl;
        exit(-1);
    }

    for (int i = 0; i < 2 * uncorr_vars; ++i) {
        csi[i] = &cs_storage[i * N_DATA];
    }
    for (int i = 0; i < uncorr_vars; ++i) {
        cs[i] = &csi[i * 2];
    }

    //Define the structure to save the candidate subspace CS_i ------ matrix [3 * N_DATA]
    double **combine, *combine_storage;
    combine_storage = (double *) malloc(N_DATA * 3 * sizeof(double));
    if (combine_storage == nullptr) {
        cout << "Malloc error on combine_storage" << endl;
        exit(-1);
    }
    combine = (double **) malloc(3 * sizeof(double *));
    if (combine == nullptr) {
        cout << "Malloc error on combine" << endl;
        exit(-1);
    }
    for (int l = 0; l < 3; ++l) {
        combine[l] = &combine_storage[l * N_DATA];
    }

    //Concatenate PC1_corr, PC2_corr
    memcpy(combine[0], newspace[0], N_DATA * sizeof(double));
    memcpy(combine[1], newspace[1], N_DATA * sizeof(double));

    free(newspace_storage);
    free(newspace);

    for (int i = 0; i < uncorr_vars; ++i) {
        //Concatenate PC1_corr, PC2_corr and i-th dimension of uncorr
        memcpy(combine[2], data[uncorr[i]], N_DATA * sizeof(double));

        PCA_transform(combine, 3, N_DATA, cs[i]);
        cout << "PCA computed on PC1_CORR, PC2_CORR and " << i+1 << "-th dimension of UNCORR" << endl;
    }

    free(combine_storage);
    free(combine);

    //boolean matrix [uncorr_var*N_DATA]: True says the data is in the circles, otherwise False
    bool **incircle, *incircle_storage;
    incircle_storage = (bool *) malloc(uncorr_vars * N_DATA * sizeof(bool));
    if (incircle_storage == nullptr) {
        cout << "Malloc error on incircle_storage" << endl;
        exit(-1);
    }
    incircle = (bool **) malloc(uncorr_vars * sizeof(bool *));
    if (incircle == nullptr) {
        cout << "Malloc error on incircle" << endl;
        exit(-1);
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        incircle[i] = &incircle_storage[i * N_DATA];
    }

    fill_n(incircle_storage, uncorr_vars * N_DATA, false);

    for (int i = 0; i < uncorr_vars; ++i) {
        cluster_report rep;
        cout << "Candidate Subspace " << i+1 << ": ";
        rep = run_K_means(cs[i], N_DATA); //Clustering through Elbow criterion on i-th candidate subspace
        for (int j = 0; j < rep.k; ++j) {
            int k = 0, previous_k = 0;
            int cls_size = cluster_size(rep, j, N_DATA);
            while (k < PERCENTAGE_INCIRCLE * cls_size) {
                double dist = 0.0;
                int actual_cluster_size = 0;
                for (int l = 0; l < N_DATA; ++l) {
                    if (rep.cidx[l] == j && !(incircle[i][l])) {
                        dist += L2distance(rep.centroids(0,j), rep.centroids(1,j), cs[i][0][l], cs[i][1][l]);
                        actual_cluster_size++;
                    }
                }
                double dist_mean = dist / actual_cluster_size;
                for (int l = 0; l < N_DATA; ++l) {
                    if (rep.cidx[l] == j && !(incircle[i][l])) {
                        if (L2distance(rep.centroids(0,j), rep.centroids(1,j), cs[i][0][l], cs[i][1][l]) <= dist_mean) {
                            incircle[i][l] = true;
                            k++;
                        }
                    }
                }
                //Stopping criterion when the data threshold is greater than remaining n_points in a cluster
                if (k == previous_k) {
                    break;
                } else {
                    previous_k = k;
                }
            }
        }
        if (save_output) {
            string fileoutname = "dataout";
            string num = to_string(i);
            string concat = fileoutname + num + ".csv";
            csv_out_info(cs[i], N_DATA, concat, incircle[i], rep);
        }
    }

    auto end = chrono::steady_clock::now();

    cout << "Outliers Identification Process: \n";

    int tot_outliers = 0;
    for (int i = 0; i < N_DATA; ++i) {
        int occurrence = 0;
        for (int j = 0; j < uncorr_vars; ++j) {
            if (!incircle[j][i]) {
                occurrence++;
            }
        }

        if (occurrence >= std::round(uncorr_vars * PERCENTAGE_SUBSPACES)) {
            tot_outliers++;
            cout << i << ") ";
            for (int l = 0; l < N_DIMS; ++l) {
                //cout << cs[0][l][i] << " ";
                cout << data[l][i] << " ";
            }
            cout << "(" << occurrence << ")\n";
        }
    }

    free(incircle_storage);
    free(incircle);
    free(cs_storage);
    free(csi);
    free(cs);
    free(data_storage);
    free(data);

    cout << "TOTAL NUMBER OF OUTLIERS: " << tot_outliers << endl;

    cout << "Elapsed time in milliseconds : "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    return 0;
}
*/

void getDatasetDims(string fname, int *dim, int *data) {
    int cols = 0;
    int rows = 0;
    ifstream file(fname);
    string line;
    int first = 1;

    while (getline(file, line)) {
        if (first) {
            istringstream iss(line);
            string result;
            while (getline(iss, result, ','))
            {
                cols++;
            }
            first = 0;
        }
        rows++;
    }
    *dim = cols;
    *data = rows;
    cout << "Dataset: #DATA = " << *data << " , #DIMENSIONS = " << *dim << endl;
    file.close();
}

void loadData(string fname, double **array, int n_dims) {
    ifstream inputFile(fname);
    int row = 0;
    while (inputFile) {
        string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            istringstream ss(s);
            while (ss) {
                for (int i = 0; i < n_dims; i++) {
                    string line;
                    if (!getline(ss, line, ','))
                        break;
                    try {
                        array[row][i] = stod(line);
                    } catch (const invalid_argument e) {
                        cout << "NaN found in file " << fname << " line " << row
                             << endl;
                        e.what();
                    }
                }
            }
        }
        row++;
    }
    if (!inputFile.eof()) {
        cerr << "Could not read file " << fname << endl;
        __throw_invalid_argument("File not found.");
    }
}

double getMean(double *arr, int n_data) {
    double sum = 0.0;

    for (int i = 0; i<n_data; i++) {
        sum += arr[i];
    }

    return sum/n_data;
}

void Standardize_dataset(double **data, int n_dims, int n_data) {

    for (int i = 0; i < n_dims; ++i) {
        double mean = getMean(data[i], n_data);
        for(int j = 0; j < n_data; j++) {
            data[i][j] = (data[i][j] - mean);
        }
    }
}

double PearsonCoefficient(double *X, double *Y, int n_data) {
    double sum_X = 0, sum_Y = 0, sum_XY = 0;
    double squareSum_X = 0, squareSum_Y = 0;

    for (int i = 0; i < n_data; i++) {
        sum_X += X[i];
        sum_Y += Y[i];
        sum_XY += X[i] * Y[i]; // sum of X[i] * Y[i]
        squareSum_X += X[i] * X[i]; // sum of square of array elements
        squareSum_Y += Y[i] * Y[i];
    }

    double corr = (double)(n_data * sum_XY - sum_X * sum_Y)
                  / sqrt((n_data * squareSum_X - sum_X * sum_X)
                         * (n_data * squareSum_Y - sum_Y * sum_Y));
    return corr;
}

void PCA_transform(double **data_to_transform, int data_dim, int n_data, double **new_space) {
    real_2d_array dset, basis;
    real_1d_array variances;
    variances.setlength(2);
    dset.setlength(n_data, data_dim);
    basis.setlength(data_dim, 2);
    for (int i = 0; i < n_data; ++i) {
        for(int j = 0; j < data_dim; j++) {
            dset[i][j] = data_to_transform[j][i];
        }
    }

    pcatruncatedsubspace(dset, n_data, data_dim, 2, 0.0, 0, variances, basis);

//    cout << "PCA result: " << endl;
//    for (int i = 0; i < data_dim; ++i) {
//        for(int j = 0; j < 2; j++) {
//            cout << basis[i][j] << " ";
//        }
//        cout << "\n";
//    }

    for (int i=0; i < n_data; i++) {
        for (int j=0;j<2;j++) {
            new_space[j][i]=0;
            for (int k=0;k<data_dim;k++) {
                new_space[j][i] += dset[i][k] * basis[k][j];
            }
        }
    }

//    cout << "The resulting subspace: " << endl;
//    for(int j = 0; j < 2; j++) {
//        for (int i = 0; i < n_data; ++i) {
//            cout << new_space[j][i] << " ";
//        }
//        cout << "\n";
//    }
}

int cluster_size(cluster_report rep, int cluster_id, int n_data) {
    int occurrence = 0;
    for (int i = 0; i < n_data; ++i) {
        if (rep.cidx[i] == cluster_id) {
            occurrence++;
        }
    }
    return occurrence;
}

cluster_report run_K_means(double **data_to_transform, int n_data) {
    mat data(2, n_data);
    mat final;
    cluster_report final_rep, previous_rep;
    previous_rep.cidx = (int *) malloc(n_data * sizeof(int));
    final_rep.cidx = (int *) malloc(n_data * sizeof(int));
    if (previous_rep.cidx == nullptr) {
        cout << "Malloc error on cidx" << endl;
        exit(-1);
    }

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < n_data; ++i) {
            data(j,i) = data_to_transform[j][i];
        }
    }

    for (int j = 1; j <= K_MAX; ++j) {
        bool status = kmeans(final, data, j, random_subset, 30, false);
        if (!status) {
            cout << "Error in KMeans run." << endl;
            exit(-1);
        }

        if (j == 1) {
            previous_rep.centroids = final;
            previous_rep.k = j;
            previous_rep.BetaCV = 0.0;
            fill_n(previous_rep.cidx, n_data, 0);
        } else {
            final_rep.centroids = final;
            final_rep.k = j;
            create_cidx_matrix(data_to_transform, n_data, final_rep);
            final_rep.BetaCV = BetaCV(data_to_transform, final_rep, n_data);
            if (abs(previous_rep.BetaCV - final_rep.BetaCV) <= ELBOW_THRES) {
                cout << "The optimal K is " << final_rep.k << endl;
                return final_rep;
            } else {
                previous_rep = final_rep;
            }
        }
    }
    cout << "The optimal K is " << final_rep.k << endl;
    return final_rep;
}

void create_cidx_matrix(double **data, int n_data, cluster_report instance) {
    for (int i = 0; i < n_data; ++i) {
        double min_dist = L2distance(instance.centroids.at(0,0), instance.centroids.at(1,0), data[0][i], data[1][i]);
        instance.cidx[i] = 0;
        for (int j = 1; j < instance.k; ++j) {
            double new_dist = L2distance(instance.centroids.at(0,j), instance.centroids.at(1,j), data[0][i], data[1][i]);
            if (new_dist < min_dist) {
                min_dist = new_dist;
                instance.cidx[i] = j;
            }
        }
    }
}

double WithinClusterSS(double **data, cluster_report instance, int n_data) {
    double wss = 0.0;
    for (int i = 0; i < n_data; ++i) {
        int cluster_idx = instance.cidx[i];
        wss += L2distance(instance.centroids.at(0,cluster_idx), instance.centroids.at(1,cluster_idx), data[0][i], data[1][i]);
    }
    return wss;
}

double BetweenClusterSS(double **data, cluster_report instance, int n_data) {
    double pc1_mean = getMean(data[0], n_data);
    double pc2_mean = getMean(data[1], n_data);
    double bss = 0.0;
    for (int i = 0; i < instance.k; ++i) {
        double n_points = cluster_size(instance, i, n_data);
        bss += n_points * L2distance(instance.centroids.at(0, i), instance.centroids.at(1, i), pc1_mean, pc2_mean);
    }

    return bss;
}

int WithinClusterPairs(cluster_report instance, int n_data) {
    int counter = 0;
    for (int i = 0; i < instance.k; ++i) {
        int n_points = cluster_size(instance, i, n_data);
        counter += (n_points - 1) * n_points;
    }
    return counter/2;
}

int BetweenClusterPairs(cluster_report instance, int n_data) {
    int counter = 0;
    for (int i = 0; i < instance.k; ++i) {
        int n_points = cluster_size(instance, i, n_data);
        for (int j = 0; j < instance.k; ++j) {
            if (i != j) {
                counter += n_points * cluster_size(instance, j, n_data);
            }
        }
    }
    return counter/2;
}

double BetaCV(double **data, cluster_report instance, int n_data) {
    double bss = BetweenClusterSS(data, instance, n_data);
    double wss = WithinClusterSS(data, instance, n_data);
    int N_in = WithinClusterPairs(instance, n_data);
    int N_out = BetweenClusterPairs(instance, n_data);

    return ((double) N_out / N_in) * (wss / bss);
}

double L2distance(double xc, double yc, double x1, double y1) {
    double x = xc - x1; //calculating number to square in next step
    double y = yc - y1;
    double dist;

    dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
    dist = sqrt(dist);

    return dist;
}

void csv_out_info(double **data, int n_data, string outdir, string name, bool *incircle, cluster_report report) {
    fstream fout;
    fout.open(outdir + name, ios::out | ios::trunc);

    for (int i = 0; i < n_data; ++i) {
        for (int j = 0; j < 2; ++j) {
            fout << data[j][i] << ",";
        }
        if (incircle[i]) {
            fout << "1,";
        } else {
            fout << "0,";
        }
        fout << report.cidx[i];
        fout << "\n";
    }

    fout.close();
    fstream fout2;
    fout2.open(outdir + "centroids_" + name, ios::out | ios::trunc);
    for (int i = 0; i < report.k; ++i) {
        fout2 << report.centroids.at(0, i) << ",";
        fout2 << report.centroids.at(1, i) << "\n";
    }
    fout2.close();
}