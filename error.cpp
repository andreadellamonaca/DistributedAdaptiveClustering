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

