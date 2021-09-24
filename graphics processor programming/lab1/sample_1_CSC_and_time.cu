#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define CSC(call) 						\
do {									\
	cudaError_t	status = call;			\
	if (status != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));			\
		exit(0);						\
	}									\
} while(0)

__global__ void kernel(int *arr, int n) {
	// threadIdx.x, threadIdx.y, threadIdx.z,
	// blockIdx.x, blockIdx.y, blockIdx.z,
	// blockDim.x, blockDim.y, blockDim.z,
	// gridDim.x, gridDim.y, gridDim.z,
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	while (idx < n) {
	//	assert(idx < n - 10);
		arr[idx] *= 2;
		idx += offset;
	}
}

int main() {
	int i, n = 10000000;
	scanf("%d", &n);
	int *arr = (int *)malloc(sizeof(int) * n);
	for(i = 0; i < n; i++)
		arr[i] = i;
	int *dev_arr;
	CSC(cudaMalloc(&dev_arr, sizeof(int) * n));

	CSC(cudaMemcpy(dev_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice));

	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

	kernel<<<256, 256>>>(dev_arr, n);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));

	printf("time = %f\n", t);

	CSC(cudaMemcpy(arr, dev_arr, sizeof(int) * n, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_arr));
	for(i = n - 10; i < n; i++)
		printf("%d ", arr[i]);
	printf("\n");
	return 0;
}