#ifndef DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
#define DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
#include <iostream>

/**
 * @file error.h
 */

using namespace std;

/**
 * @param [in] nameFunction - Name of the calling function
 * @return -1
 */
int MemoryError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -2
 */
int NullPointerError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -3
 */
int LessCorrVariablesError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -4
 */
int NoUncorrVariablesError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -5
 */
int ArgumentError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -6
 */
int DatasetReadingError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -7
 */
int PartitioningDatasetError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -8
 */
int ConversionError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -9
 */
int InputFileError(string nameFunction);
/**
 * @param [in] nameFunction - Name of the calling function
 * @return -10
 */
int MergeError(string nameFunction);

#endif //DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
