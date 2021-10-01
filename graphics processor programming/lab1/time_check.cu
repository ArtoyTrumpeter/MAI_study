#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  \
do { \
	cudaError_t state = call; \
	if (state != cudaSuccess) { \
		fprintf(stderr, "ERROR: %s:%d. Message: %s\n", __FILE__,__LINE__,cudaGetErrorString(state)); \
		exit(0); \
	} \
} while (0); \

__global__ void kernel(double* arr, long long n) {
	long long i = blockDim.x * blockIdx.x + threadIdx.x;
	long long offset = blockDim.x * gridDim.x;

	while (i < n) {
		arr[i] = abs(arr[i]);
        i += offset;
	}
}

int main() {
	long long n;
	scanf("%lld", &n);

	double* arr = (double*)malloc(n * sizeof(double));

	for(long long i = 0; i < n; i++) {
		arr[i] = -1000 + rand() % 2001;
	}

	double* dev_arr;
	CSC(cudaMalloc(&dev_arr, sizeof(double) * n));
	CSC(cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));
	
	cudaEvent_t start, end;
	for (int block = 1; block <= 1024; block *= 2) {
		for (int threads = 32; threads <= 1024; threads *= 2) {
			CSC(cudaEventCreate(&start));
			CSC(cudaEventCreate(&end));
			CSC(cudaEventRecord(start));
	
			kernel<<<block,threads>>>(dev_arr, n);
	
			CSC(cudaEventRecord(end));
			CSC(cudaEventSynchronize(end));
			float t;
			CSC(cudaEventElapsedTime(&t, start, end));
			CSC(cudaEventDestroy(start));
			CSC(cudaEventDestroy(end));

			printf("kernel = <<<%d, %d>>>, time = %f\n", block, threads, t);
		}
	}

	CSC(cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_arr));

	/*for (long long i = 0; i < n; i++) {
		printf("%.10lf ", v1[i]);
	}
	printf("\n");*/
	free(arr);
	return 0;
}