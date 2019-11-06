#include "adaptive_clustering.h"

void getDatasetDims(string fname, int *dim, int *data) {
/*************************************************************************
The algorithm calculates the number of rows and columns in a CSV file and
 saves the information into dim and data.

Input parameters:
    fname       -   the path referred to a CSV file

Output parameters:
    dim         -   the number of columns in the CSV file
    data        -   the number of rows in the CSV file
*************************************************************************/
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
/*************************************************************************
The algorithm loads a CSV file into a matrix of double.

Input parameters:
    fname       -   the path referred to a CSV file
    n_dims      -   number of columns in the CSV file.

Output parameters:
    array       -   a matrix [n_data, n_dims], in which the CSV file is loaded
*************************************************************************/
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

int cluster_size(cluster_report rep, int cluster_id, int n_data) {
/*************************************************************************
The algorithm calculates the number of elements of an indicated cluster
in the given cluster report.

Input parameters:
    rep         -   a cluster_report structure describing the K-Means
                    instance carried out.
    cluster_id  -   a number (between 0 and rep.k) indicating the cluster
                    whereby we want to know the number of data in it.
    n_data      -   number of rows in the dataset matrix.

Output result:
    an integer indicating the number of data in the cluster with the given id.
*************************************************************************/
    int occurrence = 0;
    for (int i = 0; i < n_data; ++i) {
        if (rep.cidx[i] == cluster_id) {
            occurrence++;
        }
    }
    return occurrence;
}

void create_cidx_matrix(double **data, int n_data, cluster_report instance) {
/*************************************************************************
The algorithm creates the array [1, N_DATA] which indicates the membership
of each data to a cluster through an integer.

Input parameters:
    data        -   matrix [n_data, n_dims] on which the K-Means run
                    was made.
    n_data      -   number of rows in the dataset matrix.
    instance    -   a cluster_report structure describing the K-Means
                    instance carried out.

Output parameters:
    instance    -   the input cluster_report structure updated with the
                    cidx matrix just created.
*************************************************************************/
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

double L2distance(double xc, double yc, double x1, double y1) {
/*************************************************************************
The algorithm calculates the Euclidean distance between two points
p_c and p_1.

Input parameters:
    x_c         -   the first component of p_c.
    y_c         -   the second component of p_c.
    x_1         -   the first component of p_1.
    y_1         -   the second component of p_1.

Output result:
    a double value indicating the Euclidean distance between the
    two given points
*************************************************************************/
    double x = xc - x1;
    double y = yc - y1;
    double dist;

    dist = pow(x, 2) + pow(y, 2);
    dist = sqrt(dist);

    return dist;
}

/*** Extra function to save centroids and candidate subspace in CSV files ***/
void data_out(double ****data, long *lastitem, string name, bool ***incircle, int peers, int cs, cluster_report *report) {
    fstream fout;
    fout.open("../../plot/" + name, ios::out | ios::trunc);

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
    fout2.open("../../plot/centroids_" + name, ios::out | ios::trunc);
    for (int i = 0; i < report[0].k; ++i) {
        fout2 << report[0].centroids.at(0, i) << ",";
        fout2 << report[0].centroids.at(1, i) << "\n";
    }
    fout2.close();
}