#include "error.h"

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -1
 */
int MemoryError(string nameFunction) {
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
int LessCorrVariablesError(string nameFunction) {
    cerr << nameFunction << ": Correlated dimensions must be more than 1 in order to apply PCA!" << endl;
    return -3;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -4
 */
int NoUncorrVariablesError(string nameFunction) {
    cerr << nameFunction << ": There are no candidate subspaces!" << endl;
    return -4;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -5
 */
int ArgumentsError(string nameFunction) {
    cerr << nameFunction << ": Command Line Argument Error" << endl;
    return -5;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -6
 */
int DatasetReadingError(string nameFunction) {
    cerr << nameFunction << ": Dataset Reading Error" << endl;
    return -6;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -7
 */
int PartitioningDatasetError(string nameFunction) {
    cerr << nameFunction << ": Partitioning Dataset Error" << endl;
    return -7;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -8
 */
int ConversionError(string nameFunction) {
    cerr << nameFunction << ": Conversion Error" << endl;
    return -8;
}

/**
 *
 * @param [in] nameFunction - Name of the calling function
 * @return -9
 */
int InputFileError(string nameFunction) {
    cerr << nameFunction << ": Could not read Input File!" << endl;
    return -9;
}
