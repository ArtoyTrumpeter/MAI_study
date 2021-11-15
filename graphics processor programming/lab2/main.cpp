#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cmath>
#include <algorithm>
#include "time.h"

struct uchar4{
    char x;
    char y;
    char z;
    char w;
};

int min(int a, int b) {
    return a < b ? a : b;
}

int max(int a, int b) {
    return a > b ? a : b;
}

uchar4 get_element(uchar4 *input, int w, int h, int i, int j) {
    i = max(0, min(i, w - 1));
    j = max(0, min(j, h - 1));
    return input[j * w + i];
}

double brightness(uchar4 pixel) {
	return 0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;
}

void filter(uchar4 *data, uchar4* out, int w, int h) {
	for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            double pixel1 = brightness(get_element(data, w, h, i, j));
            double pixel2 = brightness(get_element(data, w, h, i + 1, j));
            double pixel3 = brightness(get_element(data, w, h, i, j + 1));
            double pixel4 = brightness(get_element(data, w, h, i + 1, j + 1));

			double diff1 = pixel4 - pixel1;
			double diff2 = pixel2 - pixel3;

			int res = min(((int) sqrt(diff1 * diff1 + diff2 * diff2)), 255);
            uchar4 temp;
            temp.x = res;
            temp.y = res;
            temp.z = res;
            temp.w = 255;
			out[j * w + i] = temp;
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

    uchar4* out = (uchar4*) malloc(sizeof(uchar4) * w * h);
    
    clock_t begin = clock();
    filter(data, out, w, h);
    clock_t end = clock();
    double time_spent = (double) (end - begin) * 1000 / CLOCKS_PER_SEC;
    printf("%lf\n", time_spent);

	fp = fopen(output_file.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(out, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
    free(out);
	return 0;
}