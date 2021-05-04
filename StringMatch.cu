/*
This program counts the number of occurrences of a pattern in a string.
Benchmarking is done between the CPUs sequential execution and GPUs parallel execution to compare the
two approaches to solving the problem.
*/

/*
Note that there is a considerable dependency of the ratio of execution times of the CPU and GPU on the 
hardware which is being used to execute the run the program.
*/

/*
There is dependency between the threads of the GPU. Check to see if they are handled properly.
(After execution we see that it isn't working well... Have to see whats the problem)
(Try to use atomic operation)
*/

#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<string.h>

// Returns the duration from start to end times in sec
double time_elapsed(struct timespec *start, struct timespec *end) 
{
	double t;
	t = (end->tv_sec - start->tv_sec);                  // diff in seconds
	t += (end->tv_nsec - start->tv_nsec) * 0.000000001; // diff in nanoseconds
	return t;
}

// GPU function is find the total number of matches.
__global__ void GPU_Match(const char *str, const char *ptrn, int n, int m, int *count)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i > (n-m)) return;

    int j = 0;
    while(j < m && ptrn[j] == str[i+j]) j++;
    //__syncthreads();                          // Not necessary
    if(j == m) atomicAdd(count, 1);             // Doesnt work without atomic operation.... result without atomic will be wrong because of data race.

    return;
}

// CPU function is find the total number of matches.
void CPU_Match(const char *str, const char *ptrn, int n, int m, int *count)
{
    for(int i = 0; i < (n-m+1); i++)
    {
        int j = 0;
        while(j < m && ptrn[j] == str[i+j]) j++;
        if(j == m) *count = *count + 1;
    }

    return;
}

// Code execution begins here
int main()
{
    struct timespec start1, end1;   // to find GPU time
	struct timespec start2, end2;   // to find CPU time

    int n, m;
    char string[1<<20], pattern[1<<20]; // Buffers

    printf("Enter the string: ");   // Input string
    scanf(" %[^\n]", string);

    printf("Enter the pattern to be matched: ");    // Input pattern
    scanf(" %[^\n]", pattern);

    n = strlen(string);     // Find string length
    m = strlen(pattern);    // Find pattern length

    char *str, *ptrn;       // Device memory pointers
    int *count1, *count2;   // Device memory pointers


    // Allocating Device Memory
    if(cudaMallocManaged(&str, (1+n)*sizeof(char)) != cudaSuccess)
    {
        printf("Malloc Error 1!");
        return 0;
    }

    if(cudaMallocManaged(&ptrn, (1+n)*sizeof(char)) != cudaSuccess)
    {
        printf("Malloc Error 2!");
        cudaFree(str);
        return 0;
    }

    if(cudaMallocManaged(&count1, sizeof(int)) != cudaSuccess)
    {
        printf("Malloc Error 3!");
        cudaFree(str);
        cudaFree(ptrn);
        return 0;
    }

    if(cudaMallocManaged(&count2, sizeof(int)) != cudaSuccess)
    {
        printf("Malloc Error 4!");
        cudaFree(str);
        cudaFree(ptrn);
        cudaFree(count1);
        return 0;
    }

    int count = 0;

    cudaMemcpy(str, string, n, cudaMemcpyHostToDevice);                      // Copy the data to the unified memory
    cudaMemcpy(ptrn, pattern, n, cudaMemcpyHostToDevice);                    // Copy the data to the unified memory
    cudaMemcpy(count1, &count, sizeof(int), cudaMemcpyHostToDevice);         // Copy the data to the unified memory
    cudaMemcpy(count2, &count, sizeof(int), cudaMemcpyHostToDevice);         // Copy the data to the unified memory

    clock_gettime(CLOCK_REALTIME, &start1);
	GPU_Match<<<(n-m+1), 1>>>(str, ptrn, n, m, count1);   // Get GPU time
    cudaDeviceSynchronize();  
	clock_gettime(CLOCK_REALTIME, &end1);

    clock_gettime(CLOCK_REALTIME, &start2);
	CPU_Match(str, ptrn, n, m, count2);                   // Get CPU time
	clock_gettime(CLOCK_REALTIME, &end2); 

    printf("\n");
    printf("The GPU result is: %d\n", *(count1));
    printf("The CPU result is: %d\n", *(count2));
    printf("\n");
    printf("The GPU time is: %lf sec\n", time_elapsed(&start1, &end1));
    printf("The CPU time is: %lf sec\n", time_elapsed(&start2, &end2));

    cudaFree(str);
    cudaFree(ptrn);
    cudaFree(count1);
    cudaFree(count2);

    cudaDeviceReset();

    return 0;
}
