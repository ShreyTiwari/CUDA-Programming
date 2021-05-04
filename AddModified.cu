/*
Program to add two integers using the GPU instead of the CPU.
This program uses unified memory concept implemented from cuda version 6 and above. 
Terrible example to start with as the CPU can execute the opertaion 100x faster than the GPU.
Benchmarking timings to compare speeds of execution.
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

// GPU Kernel to add two numbers
__global__ void GPU_ADD(int *a, int *b)
{
	a[0] += b[0];	//add the numbers and store the result in 'a'.
}

// CPU function to add two numbers
int CPU_ADD(int a, int b)
{
	return a+b;	//return result
}

// Code execution begins here
int main()
{
	struct timespec start1, end1;	//variables to store time for GPU
	struct timespec start2, end2;	//variables to store time for CPU

	int *a1, a2;
	int *b1, b2;

	cudaMallocManaged(&a1, sizeof(int));
	cudaMallocManaged(&b1, sizeof(int));

	printf("Enter the value of a: ");	//get value of a
	scanf("%d", &a2);
	printf("Enter the value of b: ");	//get value of b
	scanf("%d", &b2);

	a1[0] = a2;
	b1[0] = b2;

	clock_gettime(CLOCK_REALTIME, &start1); //start timestamp
	GPU_ADD<<<1,1>>>(a1, b1);
	cudaDeviceSynchronize();		//wait for synchronization and to avoid bus error(core dumped) 
	int sum1 = a1[0];
	clock_gettime(CLOCK_REALTIME, &end1); //end timestamp
	
	clock_gettime(CLOCK_REALTIME, &start2); //start timestamp
	int sum2 = CPU_ADD(a2, b2);
	clock_gettime(CLOCK_REALTIME, &end2); //end timestamp

	printf("\nThe sum of the two numbers using GPU is: %d\n", sum1);
	printf("Time taken by GPU is: %E\n\n", time_elapsed(&start1, &end1));	 //print result for GPU

	printf("The sum of the two numbers using CPU is: %d\n", sum2);
	printf("Time taken by CPU is: %E\n", time_elapsed(&start2, &end2));	 //print result for CPU

	cudaFree(a1);
	cudaFree(b1);

	return 0;
}
