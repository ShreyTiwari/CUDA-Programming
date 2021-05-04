/*
Program to add two integers using the GPU instead of the CPU.
This program does not use shared memory.
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
__global__ void add(int *a, int *b)
{
	a[0] += b[0];	//add the numbers and store the result in 'a'.
}

int GPU_ADD(int a, int b)
{
	int *d_a;	//pointer to device memory
	int *d_b;	//pointer to device memory

	cudaMalloc(&d_a, sizeof(int));	//malloc space in device
	cudaMalloc(&d_b, sizeof(int));	//malloc space in device

	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);	//copy value of variable to device memory/ram
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);	//copy value of variable to device memory/ram

	add<<<1,1>>>(d_a, d_b);	//call to the gpu function with the launch configuration 

	cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);	//copy the result from device ram back to main memory

	cudaFree(d_a);	//free device space
	cudaFree(d_b);	//free device space

	return a;	//return result
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

	int a = 10000;
	int b = 10000;

	//printf("Enter the value of a: ");	//get value of a
	//scanf("%d", &a);
	//printf("Enter the value of b: ");	//get value of b
	//scanf("%d", &b);

	clock_gettime(CLOCK_REALTIME, &start1); //start timestamp
	int sum1 = GPU_ADD(a, b);
	clock_gettime(CLOCK_REALTIME, &end1); //end timestamp
	
	clock_gettime(CLOCK_REALTIME, &start2); //start timestamp
	int sum2 = CPU_ADD(a, b);
	clock_gettime(CLOCK_REALTIME, &end2); //end timestamp

	printf("\nThe sum of the two numbers using GPU is: %d\n", sum1);
	printf("Time taken by GPU is: %E\n\n", time_elapsed(&start1, &end1)); 	//print result for GPU

	printf("The sum of the two numbers using CPU is: %d\n", sum2);
	printf("Time taken by CPU is: %E\n", time_elapsed(&start2, &end2));	  	//print result for CPU

	return 0;
}
