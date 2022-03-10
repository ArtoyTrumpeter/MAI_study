#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>

using namespace std;

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)                                                          \

__global__ void histogram(int* dev_hist, int* dev_arr, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
    while (idx < n) {
		atomicAdd(dev_hist + dev_arr[idx], 1);
		idx += offset;
	}
}

__global__ void out(int* dev_hist, int* dev_arr, int* dev_out, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += offset) {
		dev_out[atomicAdd(dev_hist + dev_arr[i], -1) - 1] = dev_arr[i];
	}
}

void counting_sort(int* dev_arr, int* dev_hist, int* dev_out, int n, int max) {
	histogram<<<1024, 1024>>>(dev_hist, dev_arr, n);
	CSC(cudaGetLastError());
	
	thrust::device_ptr<int> ptr = thrust::device_pointer_cast(dev_hist);
    thrust::inclusive_scan(ptr, ptr + max + 1, ptr);
	
	out<<<1024, 1024>>>(dev_hist, dev_arr, dev_out, n);
	CSC(cudaGetLastError());
}

int maximum(int* arr, int n) {
    int max = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

int main() {
	int n;
	fread(&n, sizeof(int), 1, stdin);
	int* arr = (int*) malloc(sizeof(int) * n);
    fread(arr, sizeof(int), n, stdin);
	int max = maximum(arr, n);

    int* dev_arr;
	CSC(cudaMalloc(&dev_arr, sizeof(int) * n));
	CSC(cudaMemcpy(dev_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice));

	int* dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(int) * n));

	int* dev_hist;
	CSC(cudaMalloc(&dev_hist, sizeof(int) * (max + 1)));
	CSC(cudaMemset(dev_hist, 0, sizeof(int) * (max + 1)));

	cudaEvent_t start, end;

	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));
	
	counting_sort(dev_arr, dev_hist, dev_out, n, max);

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));
	printf("kernel = <<<64, 64>>>, time = %f\n", t);

	CSC(cudaMemcpy(arr, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost));

	//fwrite(arr, sizeof(int), n, stdout);

    CSC(cudaFree(dev_arr));
    CSC(cudaFree(dev_hist));
    CSC(cudaFree(dev_out));
	free(arr);
    
    return 0;
}