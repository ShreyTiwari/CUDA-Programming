/*
Program to find the sum of all the array elements given that the length of the array is a power of 2.
Benchmarking is been done to compare the performance of the CPU( squential with extremely powerful cores ) and
the GPU( parallel with modestly powerful cores ).
*/
/*
Sequential codes takes n-1 steps whereas parallel code takes log(n) base 2 steps.
Note: Algorithm can handle only sizes less than or equal to 2^11. 
*/
/*
Try out with shared memory?
*/ 

#include<stdio.h>
#include<time.h>
#include<cuda.h>

//returns the duration from start to end times in sec
double time_elapsed(struct timespec *start, struct timespec *end) 
{
	double t;
	t = (end->tv_sec - start->tv_sec); // diff in seconds
	t += (end->tv_nsec - start->tv_nsec) * 0.000000001; //diff in nanoseconds
	return t;
}

__global__ void GPU_Sum1(int *a, int len)   //GPU code without shared memory.
{
    int id = threadIdx.x;
    int n = blockDim.x;

    while(n > 0)
    {
        if(id < n) a[id] += a[id + n];
        __syncthreads();
        n = n/2;
    }

    return;
}

__global__ void GPU_Sum2(int *array, int len)   //GPU code with shared memory.
{
    extern __shared__ int a[];

    int id = threadIdx.x;
    int n = blockDim.x;

    a[id] = array[id];
    a[id+n] = array[id+n];
    __syncthreads();

    while(n > 0)
    {
        if(id < n) a[id] += a[id + n];
        __syncthreads();
        n = n/2;
    }

    if(id == 0) array[0] = a[0];

    return;
}

void CPU_Sum(int *a, int n)
{
    for(int i = n-1; i > 0; i--)
        a[i-1] += a[i];

    return;
}

int main()
{
    struct timespec start1, end1;	//variables to store time for GPU
    struct timespec start2, end2;	//variables to store time for GPU
    struct timespec start3, end3;	//variables to store time for CPU

    int n;      //Length of the array

    printf("Enter the value of n: ");       //Get length
    scanf("%d", &n);

    int *a1, *a2, *a3;

    if(cudaMallocManaged(&a1, n*sizeof(int)) != cudaSuccess)      //Allocate memory
    {
        printf("Malloc Error!\n");
        return 0;
    }

    if(cudaMallocManaged(&a2, n*sizeof(int)) != cudaSuccess)      //Allocate memory
    {
        printf("Malloc Error!\n");
        cudaFree(a1);
        return 0;
    }

    if(cudaMallocManaged(&a3, n*sizeof(int)) != cudaSuccess)      //Allocate memory
    {
        printf("Malloc Error!\n");
        cudaFree(a1);
        cudaFree(a2);
        return 0;
    }

    for(int i = 0; i < n; i++)      //Assign random values
    {
        a1[i] = rand()%10;
        a2[i] = a1[i];
        a3[i] = a2[i];
    }

    clock_gettime(CLOCK_REALTIME, &start1); //start timestamp
    GPU_Sum1<<<1, n/2>>>(a1, n);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &end1);	//end timestamp

    clock_gettime(CLOCK_REALTIME, &start2); //start timestamp
    GPU_Sum2<<<1, n/2, n*sizeof(int)>>>(a2, n);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &end2);	//end timestamp
        
    clock_gettime(CLOCK_REALTIME, &start3); //start timestamp
    CPU_Sum(a3, n);
    clock_gettime(CLOCK_REALTIME, &end3);	//end timestamp

    printf("\nResult of the GPU( without shared memory )  : %d\n", a1[0]);
    printf("Result of the GPU( with shared memory )     : %d\n", a2[0]);
    printf("Result of the CPU                           : %d\n", a3[0]);

    printf("\nTime taken by GPU( no shared memory ) is  : %lf\n", time_elapsed(&start1, &end1));	 //print result for GPU
    printf("Time taken by GPU( with shared memory ) is: %lf\n", time_elapsed(&start2, &end2));	 //print result for GPU
    printf("Time taken by CPU is                      : %lf\n", time_elapsed(&start3, &end3));	 //print result for CPU

    cudaFree(a1);
    cudaFree(a2);
    cudaFree(a3);

    cudaDeviceReset();

    return 0;
}
