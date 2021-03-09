#include <stdio.h>
#include <math.h>
#define A 0.0
#define B 1.0

double epsilon() {//определение машинного eps
    double eps = 1;
    while ((1 + eps) != 1) {
        eps /= 2;
    }
    return eps;
}

int main(void) {
    int numb;
    int n = 0;
    int i;
    double sum_Teylor = 1, x = 1, arg, eps = epsilon();
    scanf("%d", &numb); //ввод кол-ва делений отрезка
    double interval = (B - A) / numb;
    printf("Количество итераций\t| Значение x\t| Значение функции\t| Значение по Тейлору \t | \n");
    printf("------------------------------------------------------------------------------------------\n");
    for (arg = A; arg <= B; arg += interval) {
        sum_Teylor = 1;
        x = 1;
        for (i = 1; i <= 100; i++) {
            x *= -((arg * arg) / ((2 * i - 1) * (2 * i)));
            if (eps > fabs(x)) {
                break;
            } else {
                sum_Teylor += x;
                n++;
            }
        }
        printf(" n = %d \t\t\t| %.2lf \t\t| %.20lf \t| %.20lf \t | \n", n, arg, cos(arg), sum_Teylor);
        n = 0;
    }
    return 0;
}