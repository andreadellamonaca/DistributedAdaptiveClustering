#ifndef DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
#define DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
#include <iostream>

/**
 * @file error.h
 */

using namespace std;

int fileError(string nameFunction);
int readDatasetError(string nameFunction);
int memoryError(string nameFunction);
int partitionError(string nameFunction);
int NullPointerError(string nameFunction);
int findError(string nameFunction);
int dataError(string nameFunction);
int arithmeticError(string nameFunction);
int mergeError(string nameFunction);

#endif //DISTRIBUTEDADAPTIVECLUSTERING_ERROR_H
