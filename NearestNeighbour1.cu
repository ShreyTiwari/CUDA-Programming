/*
This program is written to find the nearest neighbour of each point in 3 deminsional
space by implementing the brute force algorithm.
The brute force approach can easily be converted into a embarassingly parallel algorithm for
the GPU where there is no interaction between the threads.
Benchmarking is done to compare the CPU and GPU computational approaches to the problem.
*/

/*
Note that there is a considerable dependency of the ratio of execution times of the CPU and GPU on the 
hardware which is being used to execute the run the program.
*/

// Importing the required headers
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<time.h>

struct position
{
    int x,y,z;      //odd number of parameters in the structure helps reducing bank conflicts in shared memory(if used).
};


// Returns the duration from start to end times in sec
double time_elapsed(struct timespec *start, struct timespec *end) 
{
	double t;
	t = (end->tv_sec - start->tv_sec); // diff in seconds
	t += (end->tv_nsec - start->tv_nsec) * 0.000000001; //diff in nanoseconds
	return t;
}

// GPU Kernel
__global__ void GPU_Find(struct position *points, int *nearest, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int min = 1<<22;
    int temp;

    if(i >= n) return;

    for(int j = 0; j < n; j++)
    {
        if(i == j) continue;

        temp = (points[i].x - points[j].x)*(points[i].x - points[j].x); 
        temp += (points[i].y - points[j].y)*(points[i].y - points[j].y);
        temp += (points[i].z - points[j].z)*(points[i].z - points[j].z);

        if(temp < min)
        {
            min = temp;
            nearest[i] = j;
        }
    }

    return;
}

// CPU Function
void CPU_Find(struct position *points, int *nearest, int n)
{
    int min;       //All the distances are going to be smaller than this.
    int temp;

    for(int i = 0; i < n; i++)
    {
        min = 1<<22;
        for(int j = 0; j < n; j++)
        {
            if(i == j) continue;

            temp = (points[i].x - points[j].x)*(points[i].x - points[j].x); 
            temp += (points[i].y - points[j].y)*(points[i].y - points[j].y);
            temp += (points[i].z - points[j].z)*(points[i].z - points[j].z);

            temp = (int)sqrt(temp);

            if(temp < min)
            {
                min = temp;
                nearest[i] = j;
            }
        }
    }

    return;
}

// Code execution begins here
int main()
{
    struct timespec start1, end1;
    struct timespec start2, end2;

    int n;

    printf("Enter the value of n: ");
    scanf("%d", &n);

    struct position *points;
    int *nearest1;
    int *nearest2;

    cudaMallocManaged(&points, n*sizeof(struct position));
    cudaMallocManaged(&nearest1, n*sizeof(int));
    cudaMallocManaged(&nearest2, n*sizeof(int));

    for(int i = 0; i < n; i++)
    {
        points[i].x = rand()%100000;
        points[i].y = rand()%100000;
        points[i].z = rand()%10000;
        nearest1[i] = -1;
        nearest2[i] = -1;
    }

    clock_gettime(CLOCK_REALTIME, &start1); //start timestamp
	GPU_Find<<<(n/128+1), 128>>>(points, nearest1, n);
	cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &end1);	//end timestamp
    
    clock_gettime(CLOCK_REALTIME, &start2); //start timestamp
	CPU_Find(points, nearest2, n);
    clock_gettime(CLOCK_REALTIME, &end2);	//end timestamp
    
    printf("\nTime taken by GPU is: %lf\n", time_elapsed(&start1, &end1));	 //print result for GPU
    printf("Time taken by CPU is: %lf\n", time_elapsed(&start2, &end2));	 //print result for CPU

    cudaFree(points);
    cudaFree(nearest1);
    cudaFree(nearest2);

    return 0;
}

/*
The results obtained by the CPU and GPU may differ. Why so?
*/
