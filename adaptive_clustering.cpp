#include "adaptive_clustering.h"
#include "error.h"

int getDatasetDims(string fname, int &dim, int &data) {
    dim = 0;
    data = 0;
    ifstream file(fname);
    
    string s;
    if(file.is_open()){
        
        while (getline(file, s)) {
           
            if (data == 0) {
                istringstream ss(s);
                string line;
                while (getline(ss, line, ',')) {
                    dim++;
                }
            }
            
            data++;
        }

    }
    else {
        cerr << "the file " << fname << " is not open!" << endl;
         return InputFileError(__FUNCTION__);
    }

    cout << "Dataset: #DATA = " << data << " , #DIMENSIONS = " << dim << endl;
    file.close();
    return 0;
}

int loadData(string fname, double **array, int n_dims) {
    if (!array) {
        return NullPointerError(__FUNCTION__);
    }
    
    ifstream inputFile(fname);
    int row = 0;
    
    string s;
    if(inputFile.is_open()){
        
        while (getline(inputFile, s)) {
            
             if (s[0] != '#') {
                 
                 istringstream ss(s);
                 
                 for (int i = 0; i < n_dims; i++) {
                     string line;
                     getline(ss, line, ',');
                        
                    try {
                        array[row][i] = stod(line);
                    }
                    catch (const invalid_argument e) {
                        cout << "NaN found in file " << fname << " line " << row << endl;
                        inputFile.close();
                        return ConversionError(__FUNCTION__);
                    }
                }
        
             }
            
             row++;
         }
        
         inputFile.close();
        
    }
    else{
        return InputFileError(__FUNCTION__);
    }
   
    return 0;
    
}

int cluster_size(cluster_report rep, int cluster_id, int n_data) {
    if (!rep.cidx) {
        exit(NullPointerError(__FUNCTION__));
    }
    
    int occurrence = 0;
    for (int i = 0; i < n_data; ++i) {
        if (rep.cidx[i] == cluster_id) {
            occurrence++;
        }
    }
    
    return occurrence;
}

int mindistCluster(mat centroids, double first_coordinate, double second_coordinate) {
    if (centroids.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }
    
    double min_dist = L2distance(centroids(0,0), centroids(1,0), first_coordinate, second_coordinate);
    int index = 0;
    
    for (int j = 1; j < centroids.n_cols; ++j) {
        double new_dist = L2distance(centroids(0,j), centroids(1,j), first_coordinate, second_coordinate);
        if (new_dist < min_dist) {
            min_dist = new_dist;
            index = j;
        }
    }
    
    return index;
}

int create_cidx_matrix(double **data, int partitionSize, cluster_report &instance) {
    if (!data || instance.centroids.is_empty()) {
        return NullPointerError(__FUNCTION__);
    }
    
    instance.cidx = (int *) calloc(partitionSize, sizeof(int));
    if (!instance.cidx) {
        return MemoryError(__FUNCTION__);
    }
    
    for (int i = 0; i < partitionSize; ++i) {
        instance.cidx[i] = mindistCluster(instance.centroids, data[0][i], data[1][i]);
    }
    
    return 0;
}

double L2distance(double xc, double yc, double x1, double y1) {
    double x = xc - x1;
    double y = yc - y1;
    double dist;

    dist = pow(x, 2) + pow(y, 2);
    dist = sqrt(dist);

    return dist;
}

int partitionData(int n_data, int peers, long **peerLastItem, long **partitionSize) {

    *peerLastItem = (long *) calloc(peers, sizeof(long));
    if (!(*peerLastItem)) {
        return MemoryError(__FUNCTION__);
    }
    *partitionSize = (long *) calloc(peers, sizeof(long));
    if (!(*partitionSize)) {
        free(*peerLastItem), *peerLastItem = NULL;
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

int computeLocalAverage(double **data, int ndims, long start, long end, double *summaries) {
    if (!data || !summaries) {
        return NullPointerError(__FUNCTION__);
    }

    double weight = 0;
    if ((end - start)) {
        weight = 1 / (double) (end - start + 1);
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
            pcc[m][l] = pcc[l][m];
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
//            if (partitionSize != 0) {
//                covarianceMatrix[i][j] = covarianceMatrix[i][j] / partitionSize;
//            }
            covarianceMatrix[j][i] = covarianceMatrix[i][j];
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

int computeLocalInitialCentroid(int centroidToSet, int partitionSize, double **data, double &actual_dist, mat &centroids) {
    if (!data || centroids.is_empty()) {
        return NullPointerError(__FUNCTION__);
    }

    actual_dist = -1.0;
    for (int dataID = 0; dataID < partitionSize; ++dataID) {
        double min_dist = L2distance(centroids(0, 0), centroids(1, 0), data[0][dataID], data[1][dataID]);
        if (min_dist == 0.0) {//the data is already a centroid
            continue;
        }
        for (int centroidID = 1; centroidID < centroidToSet; ++centroidID) {
            double dist = L2distance(centroids(0, centroidID), centroids(1, centroidID), data[0][dataID], data[1][dataID]);
            if (dist == 0.0) {//the data is already a centroid
                min_dist = 0.0;
                break;
            }
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        if (min_dist > actual_dist) {
            centroids(0, centroidToSet) = data[0][dataID];
            centroids(1, centroidToSet) = data[1][dataID];
            actual_dist = min_dist;
        }
    }

    return 0;
}

int computeLocalKMeans(long partitionSize, mat centroids, double **subspace, double *weights, double **localsum, double &error) {
    if (!subspace || !weights || !localsum || centroids.is_empty()) {
        return NullPointerError(__FUNCTION__);
    }

//    for (int l = 0; l < centroids.n_cols; ++l) {
//        weights[l] += 1;
//        localsum[0][l] += centroids(0, l);
//        localsum[1][l] += centroids(1, l);
//    }
    
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

int getCountOutliersinSubspace(int uncorr_vars, long partitionSize, bool **discarded, vector<int> &outliers) {
    if (!discarded) {
        return NullPointerError(__FUNCTION__);
    }
    outliers.clear();

    for (int k = 0; k < partitionSize; ++k) {
        int count = 0;
        for (int j = 0; j < uncorr_vars; ++j) {
            if (!discarded[j][k]) {
                count++;
            }
        }
        outliers.push_back(count);
    }

    return 0;
}