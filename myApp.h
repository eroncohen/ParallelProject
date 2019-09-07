#pragma warning(disable:4996)
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define MAX_CORDINATIONS 20
#define THREADS_PER_BLOCK 1000

typedef struct
{
	double data[MAX_CORDINATIONS];
	int classify;
} Point;

typedef struct
{
	double q;
	double alpha;
	double weights[MAX_CORDINATIONS];

} Solution;

cudaError_t calculateWithCuda(Point *points,
	unsigned int sizeOfBlock, unsigned int sizeOfThreadsPerBlock, double* weights, int K, int *numMiss);
MPI_Datatype initPointMPIType();
MPI_Datatype initSolutionMPIType();
void writeSolutionToFile(Solution solution, double QC, int K);
void sendParamsToSlaves(int* N, int* K, double* a0, int* LIMIT, int root);
void sendPointsToSlaves(int N, Point* points, int root, MPI_Datatype pointType);
int f(double* data, double* weights, int K);
void fixWeights(double* weights, int K, Point p, double alpha, int sign);
void zeroWeights(double* weights, int K);
int trainAndTest(int N, int K, int LIMIT, double alpha, double* weights, Point* points, Solution* solution);
