#include "myApp.h"

cudaError_t calculateWithCuda(Point *points,
	unsigned int sizeOfBlock, unsigned int sizeOfThreadsPerBlock, double* weights, int K, int *numMiss);

__device__ double fInGPU(double* data, double* weights, int K) {
	double res = 0;
	for (int i = 0; i < K; i++)
	{
		res += data[i] * weights[i];
	}
	res += weights[K] * 1; //for bias
	if (res >= 0)
		return 1;
	else
		return -1;
}

__global__ void calculateF(Point* points, int *results, double* weights, int K, int sizeOfThreadsPerBlock) //every thread caluclating different point misses
{
	int idOfThread = threadIdx.x;
	int idOfBlock = blockIdx.x;
	int currentId = idOfThread +
		idOfBlock*sizeOfThreadsPerBlock;
	if (fInGPU(points[currentId].data, weights, K) != points[currentId].classify)
	{
		results[currentId]++;
	}
}

__global__ void sumResults(int *counter, int *results, int size) // sum all misses
{
	*counter = 0;
	for (int i = 0; i < size; i++) {
		if (results[i] > 0) {
			(*counter)++;
		}
	}
}

cudaError_t calculateWithCuda(Point *points,
	unsigned int sizeOfBlock, unsigned int sizeOfThreadsPerBlock, double* weights, int K, int *numMiss)
{
	Point *points_cuda = 0;
	double *weights_cuda = 0;
	int *numMissGPU = 0;
	int *results = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers    .
	cudaStatus = cudaMalloc((void**)&points_cuda, sizeOfThreadsPerBlock
		* sizeOfBlock * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocPoint failed!");
		goto Error;
	}

	// Allocate GPU buffers    .
	cudaStatus = cudaMalloc((void**)&weights_cuda, (K+1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocPoint failed!");
		goto Error;
	}

	// Allocate GPU buffers   .
	cudaStatus = cudaMalloc((void**)&results, sizeOfThreadsPerBlock
		* sizeOfBlock * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers   .
	cudaStatus = cudaMalloc((void**)&numMissGPU, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(points_cuda, points, sizeOfThreadsPerBlock
		* sizeOfBlock * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(weights_cuda, weights, (K + 1) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU
	calculateF <<<sizeOfBlock, 1000 >>>(points_cuda, results, weights_cuda, K, sizeOfThreadsPerBlock);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Launch a kernel on the GPU
	sumResults <<<1, 1 >>>(numMissGPU, results, sizeOfThreadsPerBlock*sizeOfBlock);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy from GPU buffers to host memory.
	cudaStatus = cudaMemcpy(numMiss, numMissGPU, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(points_cuda);
	cudaFree(weights_cuda);

	return cudaStatus;
}