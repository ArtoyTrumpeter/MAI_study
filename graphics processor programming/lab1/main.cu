#include <stdio.h>
#include <stdlib.h>

#define CSC(call)                       \
do {                                    \
    cudaError_t status = call;          \
    if (status != cudaSuccess) {        \
        fprintf(stderr, "Error in %s: %d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));          \
        exit(0);                        \
    }                                   \
} while(0)                              \

__global__ void kernel(double *arr, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    for (idx; idx < n; idx += offset) {
        arr[idx] = abs(arr[idx]);
    }
}

int main() {
    int i, n;
    scanf("%d", &n);
    double *arr = (double*)malloc(sizeof(double) * n);
    for (i = 0; i < n; i++) {
        scanf("%lf", &arr[i]);
    }
    double *dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(double) * n));
    CSC(cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));
    kernel<<<64, 256>>>(dev_arr, n);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_arr));
    for (i = 0; i < n; i++) {
        printf("%.10f ", arr[i]);
    }
    free(arr);
    return 0;
}