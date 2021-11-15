#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <algorithm>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ uchar4 get_pixel(int i, int j) {
	uchar4 pixel = tex2D(tex, i, j);
	return pixel;
}

__device__ double brightness(uchar4 pixel) {
	return 0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;
}

__global__ void kernel(uchar4 *out, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	for (int j = idy; j < h; j += offsety) {
		for (int i = idx; i < w; i += offsetx) {
			double pixel1 = brightness(get_pixel(i, j));
			double pixel2 = brightness(get_pixel(i + 1, j));
			double pixel3 = brightness(get_pixel(i, j + 1));
			double pixel4 = brightness(get_pixel(i + 1, j + 1));

			double diff1 = pixel4 - pixel1;
			double diff2 = pixel2 - pixel3;

			int res = min(((int) sqrt(diff1 * diff1 + diff2 * diff2)), 255);
			out[j * w + i] = make_uchar4(res, res, res, 255);
		}
	}
}

int main() {
	std::string input_file, output_file;
	std::cin >> input_file >> output_file; 
	int w, h;
	FILE *fp = fopen(input_file.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));
	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;

	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

	kernel<<<dim3(16, 16), dim3(8, 32)>>>(dev_out, w, h);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	fp = fopen(output_file.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}