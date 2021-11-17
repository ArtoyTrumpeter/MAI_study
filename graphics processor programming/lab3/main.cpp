#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "time.h"

using namespace std;

enum color {R, G, B};

double dev_avg[32][3];

typedef struct {
	int x;
	int y;
} point;

typedef struct {
    char x;
    char y;
    char z;
    char w;
} uchar4;

double min_dist(uchar4* pixel, int index) {
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

void kernel(uchar4 *out, int size, int num_class) {
	for (int x = 0; x < size; x++) {
		uchar4* pixel = &out[x];
		double max_dist = 0.0;
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

	for (int i = 0; i < num_class; i++) {
		dev_avg[i][R] = 0;
		dev_avg[i][G] = 0;
		dev_avg[i][B] = 0;
		for (int j = 0; j < v[i].size(); j++) {
			point cur_point = v[i][j];
			uchar4 pixel = data[cur_point.y * w + cur_point.x];
			dev_avg[i][R] += pixel.x;
			dev_avg[i][G] += pixel.y;
			dev_avg[i][B] += pixel.z;
		}

		dev_avg[i][R] /= (double) v[i].size();
		dev_avg[i][G] /= (double) v[i].size();
		dev_avg[i][B] /= (double) v[i].size();
	}
    
    clock_t begin = clock();

	kernel(data, size, num_class);
    
    clock_t end = clock();
    double time_spent = (double) (end - begin) * 1000 / CLOCKS_PER_SEC;
    printf("%lf\n", time_spent);

	fp = fopen(output_file.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}