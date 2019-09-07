#include "myApp.h"

MPI_Datatype initPointMPIType()
{
	Point point;
	MPI_Datatype PointMPIType;
	MPI_Datatype type[2] = { MPI_DOUBLE, MPI_INT };
	int blocklen[2] = { MAX_CORDINATIONS, 1 };
	MPI_Aint disp[2];
	// Create MPI user data type for partical
	disp[0] = (char *)&point.data - (char *)&point;
	disp[1] = (char *)&point.classify - (char *)&point;
	MPI_Type_create_struct(2, blocklen, disp, type, &PointMPIType);
	MPI_Type_commit(&PointMPIType);
	return PointMPIType;
}

MPI_Datatype initSolutionMPIType()
{
	Solution solution;
	MPI_Datatype SolutionMPIType;
	MPI_Datatype type[3] = { MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE };
	int blocklen[3] = { 1,1,MAX_CORDINATIONS };
	MPI_Aint disp[3];
	// Create MPI user data type for partical
	disp[0] = (char *)&solution.q - (char *)&solution;
	disp[1] = (char *)&solution.alpha - (char *)&solution;
	disp[2] = (char *)&solution.weights - (char *)&solution;
	MPI_Type_create_struct(3, blocklen, disp, type, &SolutionMPIType);
	MPI_Type_commit(&SolutionMPIType);
	return SolutionMPIType;
}

void writeSolutionToFile(Solution solution, double QC, int K)
{
	FILE* file = fopen("C:/Users/cudauser/Desktop/CudaMPIOpenMP_onVDI/result.txt", "w");
	if (solution.q<QC)
	{
		fprintf(file, "Alpha minimum = %lf   q = %lf\n", solution.alpha, solution.q);
		for (int k = 0; k <= K; k++)
			fprintf(file, "%lf\n", solution.weights[k]);
	}
	else
		fprintf(file, "Alpha is not found\n");
}

void sendParamsToSlaves(int* N, int* K, double* a0, int* LIMIT, int root)
{
	MPI_Bcast(N, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(K, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(a0, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(LIMIT, 1, MPI_INT, root, MPI_COMM_WORLD);
}

void sendPointsToSlaves(int N, Point* points, int root, MPI_Datatype pointType)
{
	MPI_Bcast(points, N, pointType, root, MPI_COMM_WORLD);
}

int f(double* data, double* weights, int K) {
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

//the algorithm reach good q if we multiply by -1, but its not suitable the algorithm requirement (thats why its in comment)
void fixWeights(double* weights, int K, Point p, double alpha, int sign) 
{
	for (int k = 0; k < K; k++)
	{
		weights[k] = weights[k] + p.data[k] * alpha * sign/* * (-1)*/; 
	}
	weights[K] = weights[K] + alpha * sign /** (-1)*/; //bias weight
}

void zeroWeights(double* weights, int K)
{
	for (int k = 0; k <= K; k++)
	{
		weights[k] = 0;
	}
}

int trainAndTest(int N, int K, int LIMIT, double alpha, double* weights, Point* points, Solution* solution)
{
	int numMiss=0, flag, sign;
	int numMissOMP = 0, numMissCuda = 0;
	zeroWeights(weights, K);
	while (numMiss < LIMIT)
	{
		for (int i = 0; i < N; i++)
		{
			flag = 1; // to know if there was a fix weights
			sign = f(points[i].data, weights, K);
			if (sign != points[i].classify)
			{
				fixWeights(weights, K, points[i], alpha, sign);
				numMiss++;
				flag = 0;
				break;
			}
		}
		if (flag) // correct if all points classified well
			break;
	}

	//test starting
#pragma omp parallel for reduction(+:numMissOMP) //calculates 1st hlaf of the points array
	for (int i = 0; i < N/2; i++)
	{
		if (f(points[i].data, weights, K) != points[i].classify)
			numMissOMP++;
	}

	cudaError_t cudaStatus =
		calculateWithCuda(&points[N / 2],
		(N / 2) / THREADS_PER_BLOCK,
			THREADS_PER_BLOCK,weights,K, &numMissCuda); //calculates the second half with CUDA.
	if (cudaStatus != cudaSuccess) { //assure we success the operation
		fprintf(stderr, "calculateFWithCuda failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	numMiss = numMissOMP + numMissCuda;
	solution->q = (double)numMiss / (double)N;
	solution->alpha = alpha;
	for (int k = 0; k <= K; k++)
		solution->weights[k] = weights[k];

	return 0;
}

int main(int argc, char *argv[])
{
	int N, K, LIMIT;
	double a0, aMax, QC, alpha, startTime;
	double weights[MAX_CORDINATIONS];
	Point* points;
	Solution solution, temp;
	int root = 0;
	int myid, termination_tag = 10,errorCode = 999, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;
	MPI_Datatype PointMPIType = initPointMPIType();
	MPI_Datatype SolutionMPIType = initSolutionMPIType();
	if (numprocs < 2)// cannot run with less than 2 because master is not working, only managing the work
		MPI_Abort(MPI_COMM_WORLD, errorCode);
	if (myid == root) //master read data from file
	{
		//should put correct path where u put the file
		FILE* file = fopen("C:/Users/cudauser/Desktop/CudaMPIOpenMP_onVDI/data1.txt", "r");
		fscanf(file, "%d %d %lf %lf %d %lf", &N, &K, &a0, &aMax, &LIMIT, &QC);
		points = (Point*)calloc(N, sizeof(Point));

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < K; j++)
				fscanf(file, "%lf", &points[i].data[j]);
			fscanf(file, "%d", &points[i].classify);
		}
		fclose(file);
		solution.alpha = aMax + 1; //update high alpha to reach better solution
		solution.q = QC; //update high q to reach better solution
		startTime = MPI_Wtime();// starting to measure the algorithm runtime
	}
	sendParamsToSlaves(&N, &K, &a0, &LIMIT, root); // master sending relevant parameters to slaves
	if (myid != root) //slaves allocate memory of points and weights
	{
		points = (Point*)calloc(N, sizeof(Point));
	}
	sendPointsToSlaves(N, points, root, PointMPIType); // master sending all points to slaves

	if (myid == root)
	{
		int counter = 0;// counting how much slaves got termination tag
		int cont = 1; // boolean to know if a good q is reached
		alpha = a0;
		for (int i = 1; i < numprocs; i++) // master sending 1 alpha for each slave firstable
		{
			if (alpha < aMax) // wrong if number of slaves is bigger than number of alphas
			{
				MPI_Send(&alpha, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); // sending alpha to slave
				alpha += a0;
			}
			else
			{
				counter++; // another slave got termination tag
				MPI_Send(&alpha, 1, MPI_DOUBLE, i, termination_tag, MPI_COMM_WORLD); // sending termination tag to slave
			}
		}

		while (counter < numprocs - 1)
		{
			MPI_Recv(&temp, 1, SolutionMPIType, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			if (temp.q < QC || alpha>aMax || cont == 0) //master checks if the current q is good enough or other q is reached before
			{
				if (temp.alpha < solution.alpha && temp.q<QC)
				{
					solution = temp;
					cont = 0;
				}
				MPI_Send(&alpha, 1, MPI_DOUBLE, status.MPI_SOURCE, termination_tag, MPI_COMM_WORLD);// sending termination tag to slave
				counter++; // another slave got termination tag
			}
			else
			{
				MPI_Send(&alpha, 1, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD); // sending another alpha to check
				alpha += a0;
			}
		}
		writeSolutionToFile(solution, QC, K);
	}
	else
	{
		int terminate = 0; // to know if cuda doesn't work in trainAndTest function
		while (1) //slaves work until they get a termination tag
		{
			MPI_Recv(&alpha, 1, MPI_DOUBLE, root, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == termination_tag) // correct if slaves got termination tag
			{
				break;
			}
			else
			{
				terminate = trainAndTest(N, K, LIMIT, alpha, weights, points, &solution); // all algorithm of train and test
				if (terminate)
					return 1;
				MPI_Send(&solution, 1, SolutionMPIType, root, 0, MPI_COMM_WORLD);
			}
		}
	}
	if (myid == root)
		printf("runtime = %lf seconds\n", MPI_Wtime() - startTime);
	free(points);

	MPI_Finalize();
	return 0;
}
