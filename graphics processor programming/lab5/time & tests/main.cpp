#include <iostream>
#include "time.h"

using namespace std;

void counting_sort(int* table, int* out, int* count_table, int size, int max) {
    for (int i = 0 ; i < (max + 1); i++) {
        count_table[i] = 0;
    }
    for (int i = 0 ; i < size; i++) {
        count_table[table[i]]++;
    }
    for (int i = 1 ; i < (max + 1); i++) {
        count_table[i] = count_table[i] + count_table[i - 1];
    }
    for (int i = (size - 1); i >= 0; i--) {
        out[--count_table[table[i]]] = table[i];
    }
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
    int size;
    cin >> size;
    int* table = (int*) malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++) {
        table[i] = rand() % 1001;
    }
    int max = maximum(table, size);
    int* out = (int*) malloc(sizeof(int) * size);
    int* count_table = (int*) malloc(sizeof(int) * (max + 1));
    
    clock_t begin = clock();
    counting_sort(table, out, count_table, size, max);
    clock_t end = clock();
    double time_spent = (double) (end - begin) * 1000 / CLOCKS_PER_SEC;
    printf("%lf\n", time_spent);
    
    free(table);
    free(out);
    return 0;
}