/*
The following program is written to add two matrices.
Benchmarking the time of execution on the CPU and the GPU is done to compare the sequential and parallel
approach to solve this problem.
*/
/*
Note that there is a considerable dependency of the ratio of execution times of the CPU and GPU on the 
hardware which is being used to execute the run the program.
*/

#include<stdio.h>
#include<cuda.h>
#include<time.h>

//returns the duration from start to end times in sec
double time_elapsed(struct timespec *start, struct timespec *end) 
{
	double t;
	t = (end->tv_sec - start->tv_sec); // diff in seconds
	t += (end->tv_nsec - start->tv_nsec) * 0.000000001; //diff in nanoseconds
	return t;
}

__global__ void GPU_ADD(int **a, int **b, int **c, int n)
{
    int i = blockIdx.x / n;
    int j = blockIdx.x % n;

    c[i][j] = a[i][j] + b[i][j];

    return;
}

void CPU_ADD(int **a, int **b, int **c, int n)
{
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            c[i][j] = a[i][j] + b[i][j];

    return;
}

int main()
{
	struct timespec start1, end1;	//variables to store time for GPU
    struct timespec start2, end2;	//variables to store time for CPU

    int n;

    printf("Enter the value of n: ");
    scanf("%d", &n);

    int **a;
    int **b;
    int **c;

    if(cudaMallocManaged(&a, n*sizeof(int*)) != cudaSuccess)
    {
        printf("Allocation for A* failed");
        return 0;
    }

    if(cudaMallocManaged(&b, n*sizeof(int*)) != cudaSuccess)
    {
        printf("Allocation for B* failed");
        cudaFree(a);
        return 0;
    }

    if(cudaMallocManaged(&c, n*sizeof(int*)) != cudaSuccess)
    {
        printf("Allocation for C* failed");
        cudaFree(a);
        cudaFree(b);
        return 0;
    }

    for(int i = 0; i < n; i ++)
    {
        cudaMallocManaged(&(a[i]), n*sizeof(int));
        cudaMallocManaged(&(b[i]), n*sizeof(int));
        cudaMallocManaged(&(c[i]), n*sizeof(int));
    }

    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
        {
            a[i][j] = 1;
            b[i][j] = 2;
            c[i][j] = 0;
        }

    clock_gettime(CLOCK_REALTIME, &start1); //start timestamp
	GPU_ADD<<<(n*n), 1>>>(a, b, c, n);
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
	CPU_ADD(a, b, c, n);
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

    for(int i = 0; i < n; i++)  //Free allocated space
    {
        cudaFree(a[i]);
        cudaFree(b[i]);
        cudaFree(c[i]);
    }

    cudaFree(a);    //Free pointers
    cudaFree(b);
    cudaFree(c);
    
    return 0;
}