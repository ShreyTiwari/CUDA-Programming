/*
Program to multiply two matrices using the CPU and the GPU.
Benchmarking the timings to compare CPUs sequential execution to the parallel computation
capability of a GPU.
*/

/*
Note that there is a considerable dependency of the ratio of execution times of the CPU and GPU on the 
hardware which is being used to execute the run the program.
*/

// Importing the required headers
#include<stdio.h>
#include<cuda.h>
#include<time.h> 

// Returns the duration from start to end times in sec
double time_elapsed(struct timespec *start, struct timespec *end) 
{
	double t;
	t = (end->tv_sec - start->tv_sec); // diff in seconds
	t += (end->tv_nsec - start->tv_nsec) * 0.000000001; //diff in nanoseconds
	return t;
}

// GPU Kernel
__global__ void GPU_MUL(int **a, int **b, int **c, int n)
{
	int i = blockIdx.x;		//each block operates on one row a/c.
	int j = threadIdx.x;	//each thread of a block operates on one column of the b/c. 
	
	int ans = 0;
	for(int k = 0; k < n; k++)		//once you know the the row of a and column of b, just multiply.
		ans += a[i][k] * b[k][j];

	c[i][j] = ans;		//store result.

	return;
}

/*
// CPU Function
void CPU_MUL(int **a, int **b, int **c, int n)
{
	int ans;

	for(int i = 0; i < n; i++)		//General way of multiplying matrices in c.
	{
		for(int j = 0; j < n; j++)
		{
			ans = 0;
			for(int k = 0; k < n; k++)
			{
				ans += a[i][k] * b[k][j];
			}
			c[i][j] = ans;
		}
	}

	return;
}
*/

// CPU Function
void CPU_MUL(int **a, int **b, int **c, int n)
{
	int ans;
	for(int i = 0; i < n; i++)		//initialise to zero.
	{
		for(int k = 0; k < n; k++)
		{
			c[i][k]=0;
		}
	}

	for(int i = 0; i < n; i++)		//efficient way of multiplying matrices considering memory and caches.
	{
		for(int k = 0; k < n; k++)
		{
			for(int j = 0; j < n; j++)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return;
}

// Code execution begins here
int main()
{
	struct timespec start1, end1;	//variables to store time for GPU
	struct timespec start2, end2;	//variables to store time for CPU

	int n;
	
	printf("Enter the value of n: ");
	scanf("%d", &n);				//get value of n.

	int **a;
	int **b;
	int **c;
	
	cudaMallocManaged(&a, n*sizeof(int*));	// pointer to allocated shared memory
	cudaMallocManaged(&b, n*sizeof(int*));	// pointer to allocated shared memory
	cudaMallocManaged(&c, n*sizeof(int*));	// pointer to allocated shared memory

	for(int i = 0; i < n; i++)
	{
		cudaMallocManaged(&(a[i]), n*sizeof(int));	// allocate shared memory
		cudaMallocManaged(&(b[i]), n*sizeof(int));	// allocate shared memory
		cudaMallocManaged(&(c[i]), n*sizeof(int));	// allocate shared memory
	}
	
	for(int i = 0; i < n; i++)		// matrix A is the identity and matrix B has only 1's as its entries 
		for(int j = 0; j < n; j++)	
		{
			if(i == j) a[i][j] = 1;
			else a[i][j] = 0;
			b[i][j] = 1;
			c[i][j] = 0;
		}

	
	clock_gettime(CLOCK_REALTIME, &start1); //start timestamp
	GPU_MUL<<<n, n>>>(a, b, c, n);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &end1);	//end timestamp

	printf("\nResult of the GPU is :\n");	//print the result
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			printf("%d ", c[i][j]);
		}
		printf("\n");
	}


	clock_gettime(CLOCK_REALTIME, &start2); //start timestamp
	CPU_MUL(a, b, c, n);
	clock_gettime(CLOCK_REALTIME, &end2); 	//end timestamp

	printf("\nResult of the CPU is :\n");	//print the result
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			printf("%d ", c[i][j]);
		}
		printf("\n");
	}

	printf("\nTime taken by GPU is: %lf\n", time_elapsed(&start1, &end1));	 //print time for GPU
	printf("Time taken by CPU is: %lf\n", time_elapsed(&start2, &end2));	 //print time for CPU

	for(int i = 0; i < n; i++)	//free the allocated memory space
	{
		cudaFree(a[i]);
		cudaFree(b[i]);
		cudaFree(c[i]);
	}
	
	cudaFree(a);	//free the pointers to the allocated memory space.
	cudaFree(b);
	cudaFree(c);

	cudaDeviceReset();

	return 0;
}
