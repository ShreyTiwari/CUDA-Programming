/*
Program to add two long integer integer arrays.
Hopefully this is a better exploitation of the GPU's parallel computing capabilities.
Benchmarking timing to compare execution speeds.
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
__global__ void GPU_ADD(int *a, int *b, int n)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if(id > n) return;
	else b[id] += a[id];
}

/*
// GPU Kernel Varient
__global__ void GPU_ADD(int *a, int *b, int n)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	for(int i = index; i < n; i += stride)
		b[i] += a[i];
}
*/

// CPU function
void CPU_ADD(int *a, int *b, int n)
{
	for(int i = 0; i < n; i++)
		b[i] += a[i];
}

// Code execution begins here
int main()
{
	struct timespec start1, end1;	//variables to store time for GPU
	struct timespec start2, end2;	//variables to store time for CPU
	
	int n = 1<<20;
	int *a1, *b1, *a2, *b2;
	
	cudaMallocManaged(&a1, n*sizeof(int));
	cudaMallocManaged(&a2, n*sizeof(int));
	cudaMallocManaged(&b1, n*sizeof(int));
	cudaMallocManaged(&b2, n*sizeof(int));

	for(int i = 0; i < n; i++)
	{
		a1[i] = a2[i] = 1;
		b1[i] = b2[i] = 2;
	}

	clock_gettime(CLOCK_REALTIME, &start1); //start timestamp
	GPU_ADD<<<1024, 1024>>>(a1, b1, n);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &end1); //end timestamp
	
	clock_gettime(CLOCK_REALTIME, &start2); //start timestamp
	CPU_ADD(a2, b2, n);
	clock_gettime(CLOCK_REALTIME, &end2); //end timestamp

	cudaFree(a1);
	cudaFree(a2);
	cudaFree(b1);
	cudaFree(b2);

	printf("Time taken by GPU is: %lf\n", time_elapsed(&start1, &end1));	 //print result for GPU
	printf("Time taken by CPU is: %lf\n", time_elapsed(&start2, &end2));	 //print result for CPU

	return 0;
}
