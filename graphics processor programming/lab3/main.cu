#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#define N_INF_FLOAT __int_as_float(0xff800000)

using namespace std;

enum color {R, G, B};

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

typedef struct{
	int x;
	int y;
} point;

__constant__ double dev_avg[32][3];

__device__ double min_dist(uchar4* pixel, int index) {
	double result = 0.0;
	double diff[3];
	diff[R] = (double) pixel->x - dev_avg[index][R];
	diff[G] = (double) pixel->y - dev_avg[index][G];
	diff[B] = (double) pixel->z - dev_avg[index][B];

	for (int i = 0; i < 3; i++) {
		result += diff[i] * diff[i];
	}
	result *= -1.0;
	return result;

}

__global__ void kernel(uchar4 *out, int size, int num_class) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int x = idx; x < size; x += offsetx) {
		uchar4* pixel = &out[x];
		double max_dist = N_INF_FLOAT;
		int maxInd = 0;
		for (int y = 0; y < num_class; y++) {
			double temp_dist = min_dist(pixel, y);
			if (temp_dist > max_dist) {
				max_dist = temp_dist;
				maxInd = y;
			}
		}
		out[x].w = (unsigned char) maxInd;
    }
		
}

int main() {
    string input_file, output_file;
    int num_class, num_pair, w, h, size;
    cin >> input_file >> output_file >> num_class;
    
	vector< vector <point> > v(num_class);
	for (int i = 0; i < num_class; i++) {
		cin >> num_pair;
		v[i].resize(num_pair);
		for (int j = 0; j < num_pair; j++) {
			cin >> v[i][j].x >> v[i][j].y;
		}
	}
	
	FILE *fp = fopen(input_file.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	size = w * h;
	double avg[32][3];

	for (int i = 0; i < num_class; i++) {
		avg[i][R] = 0;
		avg[i][G] = 0;
		avg[i][B] = 0;
		for (int j = 0; j < v[i].size(); j++) {
			point cur_point = v[i][j];
			uchar4 pixel = data[cur_point.y * w + cur_point.x];
			avg[i][R] += pixel.x;
			avg[i][G] += pixel.y;
			avg[i][B] += pixel.z;
		}

		avg[i][R] /= (double) v[i].size();
		avg[i][G] /= (double) v[i].size();
		avg[i][B] /= (double) v[i].size();
	}

	uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));
	CSC(cudaMemcpy(dev_out, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
	CSC(cudaMemcpyToSymbol(dev_avg, avg, sizeof(double) * 32 * 3));

	kernel<<<8, 16>>>(dev_out, size, num_class);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	CSC(cudaFree(dev_out));

	fp = fopen(output_file.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}
