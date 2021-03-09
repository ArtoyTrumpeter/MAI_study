#include <stdio.h>


int main(void)
{
    int first_point, count, max_size, size, i, j, k, num = 0, line = 0, max_line = 0;
    scanf("%d%d", &count, &max_size);
    int A[max_size][max_size];
    int Answer[count][max_size * max_size];
    for (int z = 0; z < count; z++) {
        scanf("%d\n", &size);
        for (int N = 0; N < size; N++) {
            for (int M = 0; M < size; M++) {
                scanf("%d", &(A[N][M]));
            }
        }
        first_point = (size - 1) / 2;
        Answer[num][line] = A[first_point][first_point];
        line++;
        if (line > max_line) {
            max_line = line;
        }
        i = first_point;
        j = first_point;
        k = 1;
        if ((size % 2) == 0) {
            while (j != (size - 1)) {
                for (int l = 0; l < k; l++) {
                    i++;
                    Answer[num][line] = A[i][j];
                    line++;
                }
                for (int m = 0; m < k; m++) {
                    j++;
                    Answer[num][line] = A[i][j];
                    line++;
                }
                if ((i == (size - 1)) && (j == (size - 1))) {
                    for (int n = 0; n < k; n++) {
                        i--;
                        Answer[num][line] = A[i][j];
                        line++;
                    }
                } else {
                    for (int o = 0; o < (k + 1); o++) {
                        i--;
                        Answer[num][line] = A[i][j];
                        line++;
                    }
                }
                if (j != (size - 1)) {
                    for (int p = 0; p < (k + 1); p++) {
                        j--;
                        Answer[num][line] = A[i][j];
                        line++;
                    }
                }
                k = k + 2;
            }
        } else {
            while ((i != 0) && (j != 0)) {
                for (int q = 0; q < k; q++) {
                    i++;
                    Answer[num][line] = A[i][j];
                    line++;
                }
                for (int s = 0; s < k; s++) {
                    j++;
                    Answer[num][line] = A[i][j];
                    line++;
                }
                for (int t = 0; t < (k + 1); t++) {
                    i--;
                    Answer[num][line] = A[i][j];
                    line++;
                }
                for (int u = 0; u < (k + 1); u++) {
                    j--;
                    Answer[num][line] = A[i][j];
                    line++;
                }
                k = k + 2;
            }
            for (int r = 0; r < (k - 1); r++) {
                i++;
                Answer[num][line] = A[i][j];
                    line++;
            }
        }
        num++;
        if (line > max_line) {
            max_line = line;
        }
        line = 0;
    }
    for (int lenght = 0; lenght < num; lenght++) {
        printf("\n");
        for (int straight = 0; straight < max_line; straight++) {
            printf("%d ", Answer[lenght][straight]);
        }
    }
    return 0;
}
