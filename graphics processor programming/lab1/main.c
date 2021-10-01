#include "stdio.h"
#include "stdlib.h"
#include "time.h"

int main() {
    long long n;
    scanf("%lld", &n);
    double* arr = (double*) malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        arr[i] = -1000 + rand() % 2001;
    }
    clock_t begin = clock();
    for (int i = 0; i < n; i++) {
        arr[i] = abs(arr[i]);
    }
    clock_t end = clock();
    double time_spent = (double) (end - begin) * 1000 / CLOCKS_PER_SEC;
    printf("%lf\n", time_spent);
    free(arr);
    return 0;
}