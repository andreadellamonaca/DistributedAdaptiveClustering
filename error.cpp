#include "error.h"

int MemoryError(string nameFunction) {
    cerr << nameFunction << ": Not enough memory" << endl;
    return -1;
}

int NullPointerError(string nameFunction) {
    cerr << nameFunction << ": NullPointer Error" << endl;
    return -2;
}

int LessCorrVariablesError(string nameFunction) {
    cerr << nameFunction << ": Correlated dimensions must be more than 1 in order to apply PCA!" << endl;
    return -3;
}

int NoUncorrVariablesError(string nameFunction) {
    cerr << nameFunction << ": There are no candidate subspaces!" << endl;
    return -4;
}

int ArgumentsError(string nameFunction) {
    cerr << nameFunction << ": Command Line Argument Error" << endl;
    return -5;
}

int DatasetReadingError(string nameFunction) {
    cerr << nameFunction << ": Dataset Reading Error" << endl;
    return -6;
}

int PartitioningDatasetError(string nameFunction) {
    cerr << nameFunction << ": Partitioning Dataset Error" << endl;
    return -7;
}

int ConversionError(string nameFunction) {
    cerr << nameFunction << ": Conversion Error" << endl;
    return -8;
}

int InputFileError(string nameFunction) {
    cerr << nameFunction << ": Could not read Input File!" << endl;
    return -9;
}

int MergeError(string nameFunction) {
    cerr << nameFunction << ": MergeError" << endl;
    return -10;
}
