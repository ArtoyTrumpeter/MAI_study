#include <iostream>
#include <vector>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define EPS 1e-7

using namespace std;

#define CSC(call)  												    \
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

struct comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

void print_the_answer(double* AB, int n, int m, int k, vector<int> steps) {
    cout.precision(10);
    cout.setf(ios::scientific);
    int step_id = 0;
    for (int i = 0; i < m; i++) {
        if ((steps.size() != 0) && (steps[step_id] == i) && (step_id < steps.size())) {
            for (int j = m; j < (m + k - 1); j++) {
                cout << AB[j * n + step_id] / AB[i * n + step_id] << " ";
            }
            cout << AB[(m + k - 1) * n + step_id] / AB[i * n + step_id];
            step_id++;
        } else {
            for (int j = 0; j < (k - 1); j++) {
                cout << 0.0 << " ";
            }
            cout << 0.0;
        }
        cout << "\n";
    }
}

__global__ void swap(double* matrix, int rows, int cols, int p, int q, int current_i) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    double temp;
    for (int i = idx + current_i; i < rows; i += offsetx) {
        temp = matrix[i * cols + p];
        matrix[i * cols + p] = matrix[i * cols + q];
        matrix[i * cols + q] = temp;
    }
}

__global__ void down_pass(double* matrix, int rows, int cols, int i, int i1) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int p = 1 + i + idy; p < rows; p += offsety) {
        for (int q = 1 + i1 + idx; q < cols; q += offsetx) {
            matrix[p * cols + q] -= matrix[p * cols + i1] * matrix[i * cols + q] / matrix[i * cols + i1];
        }
    }
}

__global__ void up_pass(double* matrix, int n, int m, int k, int steps_row, int steps_num) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int p = m + idy; p < (m + k); p += offsety) {
        for (int q = idx; q < steps_num; q += offsetx) {
            matrix[p * n + q] -= matrix[p * n + steps_num] * matrix[steps_row * n + q] 
            / matrix[steps_row * n + steps_num];;
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, k;
    cin >> n >> m >> k;
    const int size_AB = n * (m + k);
    double* AB = (double*) malloc(sizeof(double) * size_AB);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AB[j * n + i] = -1000 + rand() % 2001;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = m; j < (m + k); j++) {
            AB[j * n + i] = -1000 + rand() % 2001;
        }
    }

    comparator comp;
    vector<int> steps;

    double *dev_AB;
    CSC(cudaMalloc(&dev_AB, sizeof(double) * size_AB));
    CSC(cudaMemcpy(dev_AB, AB, sizeof(double) * size_AB, cudaMemcpyHostToDevice));

    thrust::device_ptr<double> AB_ptr;
    AB_ptr = thrust::device_pointer_cast(dev_AB);

    cudaEvent_t start, end;
    
    CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

    for (int i = 0, j = 0; (i < m) && (j < n); i++) {
        auto start_pos = AB_ptr + i * n + j;
        auto end_pos = AB_ptr + (i + 1) * n;
        auto result = thrust::max_element(start_pos, end_pos, comp);
        int max = result - start_pos;
        double main_element = fabs(*result);
        if (main_element < EPS) {
            continue;
        }
        steps.push_back(i);
        if (max != 0) {
            swap<<<16, 16>>>(dev_AB, m + k, n, j, j + max, i);
            CSC(cudaGetLastError());
        }
        down_pass<<<dim3(32, 32), dim3(32, 32)>>>(dev_AB, m + k, n, i, j);
        CSC(cudaGetLastError());
        j++;
    }

    int idx = steps.size() - 1;
    for (auto it = steps.rbegin(); it != steps.rend(); it++, idx--) {
        up_pass<<<dim3(32, 32), dim3(32, 32)>>>(dev_AB, n, m, k, *it, idx);
        CSC(cudaGetLastError());
    }

    CSC(cudaEventRecord(end));
    CSC(cudaEventSynchronize(end));
    float t;
    CSC(cudaEventElapsedTime(&t, start, end));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(end));
    printf("kernel = <<<32, 32>>>, time = %f\n", t);

    CSC(cudaMemcpy(AB, dev_AB, sizeof(double) * size_AB, cudaMemcpyDeviceToHost));
    //print_the_answer(AB, n, m, k, steps);

    CSC(cudaFree(dev_AB));
    free(AB);
    return 0;
}