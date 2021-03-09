#include <stdio.h>


int main(void)
{
    int first_point, n, max_size, size, i, j, k;
    scanf("%d%d", &n, &max_size);
    int A[max_size] [max_size];
    for (int m = 0; m < n; m++) {
        scanf("%d\n", &size);
        for (int N = 0; N < size; N++) {
            for (int M = 0; M < size; M++) {
                scanf("%d", &(A[N] [M]));
            }  
        }
        first_point = (size - 1) / 2;
    	printf("%d ", A[first_point] [first_point]);
    	i = first_point;
    	j = first_point;
        k = 1;
    	if ((size % 2) == 0) {
            while ((i != 0) && (j != (size - 1))) {
    	        for (int o = 0; o < k; o++) {
                    i++;
                    printf("%d ", A[i] [j]);
                }
                for (int p = 0; p < k; p++) {
                    j++;
                    printf("%d ", A[i] [j]);
                }
                if ((i == (size - 1)) && (j == (size - 1))) { 
                    for (int x = 0; x < k; x++) {
                        i--;
                        printf("%d ", A[i] [j]);
                    }
                } else {
                    for (int q = 0; q < (k + 1); q++) {
                        i--;
                        printf("%d ", A[i] [j]);
                    }
                }
                if ((i != 0) && (j != (size - 1))) {
                    for (int r = 0; r < (k + 1); r++) {
                        j--;
                        printf("%d ", A[i] [j]);
                    }
                }
                k = k + 2;
            }
    	} else {
            while ((i != (size - 1)) && (j != 0)) {
                if ((i == 0) && (j == 0)) {
                    for (int w = 0; w < (k - 1); w++) {
                        i++; 
                        printf("%d ", A[i] [j]);
                    }
                } else { 
                    for (int s = 0; s < k; s++) {
                        i++; 
                        printf("%d ", A[i] [j]);
                    }
                }
                if ((i != (size - 1)) && (j != 0)) {
                    for (int t = 0; t < k; t++) {
                        j++;
                        printf("%d ", A[i] [j]);
                    }
                    for (int u = 0; u < (k + 1); u++) {
                        i--;
                        printf("%d ", A[i] [j]);
                    }
                    for (int v = 0; v < (k + 1); v++) {
                        j--;
                        printf("%d ", A[i] [j]);
                    }
                }
                k++;
            }
        }
        printf("\n");
    }
    return 0;
}
