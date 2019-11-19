#ifndef DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
#define DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
#include <iostream>

/**
 * @file error.h
 */

using namespace std;

int MemoryError(string nameFunction);
int NullPointerError(string nameFunction);
int LessCorrVariablesError(string nameFunction);
int NoUncorrVariablesError(string nameFunction);
int ArgumentsError(string nameFunction);
int DatasetReadingError(string nameFunction);
int PartitioningDatasetError(string nameFunction);
int ConversionError(string nameFunction);
int InputFileError(string nameFunction);
int MergeError(string nameFunction);

#endif //DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
