#include "error.h"

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -1
 */
int memoryError(string nameFunction) {
    cerr << nameFunction << ": Not enough memory" << endl;
    return -1;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -2
 */
int NullPointerError(string nameFunction) {
    cerr << nameFunction << ": NullPointer Error" << endl;
    return -2;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -3
 */
int dataError(string nameFunction) {
    cerr << nameFunction << ": Corrupt data Error" << endl;
    return -3;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -4
 */
int arithmeticError(string nameFunction) {
    cerr << nameFunction << ": Arithmetic Error" << endl;
    return -4;
}


/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -7
 */
int fileError(string nameFunction) {
    cerr << nameFunction << ": Can't open file: " << "clustered.csv" << endl;
    return -7;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -8
 */
int readDatasetError(string nameFunction) {
    cerr << nameFunction << ": Can't load dataset: " << "clustered.csv" << endl;
    return -8;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -9
 */
int partitionError(string nameFunction) {
    cerr << nameFunction << ": Error partition dataset" << endl;
    return -9;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -10
 */
int findError(string nameFunction) {
    cerr << nameFunction << ": Find Error" << endl;
    return -10;
}

int mergeError(string nameFunction) {

    cerr << nameFunction << ": Find Error" << endl;
    return -11;
}


