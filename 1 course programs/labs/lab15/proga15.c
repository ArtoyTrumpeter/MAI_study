#include <stdio.h>


int main(void)
{
    int max_size, n, max, count, z, size;
    scanf("%d%d", &n, &max_size);
    int A[max_size][max_size];
    int B[max_size];
    for (int m = 0; m < n; m++) {
        scanf("%d\n", &size);
        for (int N = 0; N < size; N++) {
            for (int M = 0; M < size; M++) {
                scanf("%d", &(A[N][M]));
            }
        }
        max = size;
        for (int a = 0; a < (max - 1); a++) {
            for (int b = 0; b < size; b++) {
                B[b] = A[a][b];
            }
            count = 0;
            for (int l = 1; l < (max - a); l++) {
                for (int j = 0; j < size; j++) {
                    if (B[j] != A[a + l][j]) {
                        break;
                    } else if (j == (size - 1)) {
                        A[a + l][j] = 0;
                    }
                }
            }
            for (int col = 0; col < max; col++) {
                if (A[col][size - 1] == 0) {
                    count++;
                }
            }
            z = A[max - 1][size - 1];
            for (int c = 1; c < max; c++) {
                for (int d = 0; d < size; d++) {
                    if ((c != (max - 1)) && (A[c][size - 1] == 0)) {
                        A[c][d] = A[c + 1][d];
                        A[c + 1][size - 1] = 0;
                    } else if (c == (max - 1)) {
                        A[max - 1][size - 1] = z;
                    }
                }
            }
            for (int col1 = 0; col1 < count; col1++) {
                max--;
            }
        }
        for (int N = 0; N < max; N++) {
            for (int M = 0; M < size; M++) {
                printf("%d ", A[N][M]);
            }
            printf("\n");
        }
    }
    return 0;
}