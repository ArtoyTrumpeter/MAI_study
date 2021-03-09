#include <stdio.h>
#include <math.h>

double mainf4(double x ){
    return 3 * x - 14 + exp(x) - exp(- x);
}

double mainf5(double x) {
    return sqrt(1 - x) - tan(x);
}

double f4(double x) {
    return log(14 + exp(- x) - 3 * x);
}

double f5(double x) {
    return atan(sqrt(1 - x));
}

double derivativef4(double x) {
    return 3 + exp(x) + exp(- x);
}

double derivativef5(double x) {
    return -1/(2 * sqrt(1 - x )) - 1/(cos(x) * cos(x));
}

double epsilon() {
    double eps = 1.0;
    while (1.0 + (eps / 2.0) > 1.0) {
        eps /= 2;
    }
    return eps;
}

double dht(double left, double right, double (*f)(double)) {
    double eps = epsilon();
    while (fabs(f(left) * f(right)) >= eps) {
        if (f(left) * f((left + right) / 2.0) > 0) {
            left = (left + right) / 2.0;
        } else {
            right = (left + right) / 2.0;
        }
    }
    return (left + right) / 2.0;
}

double iter(double left, double right, double (*f)(double)) {
    double x, x0, eps = epsilon();
    x0 = (left + right) / 2.0;
    x = right;
    while (fabs(x0 - x) >= eps) {
        x0 = x;
        x = f(x0);
    }
    return x;
}

double newton(double left, double right, double (*f)(double), double (*df)(double)) {
    double x, eps = epsilon();
    x = (left + right) / 2.0;
    while (fabs(f(x) / df(x)) >= eps) {
        x = x - (f(x) / df(x));
    }
    return x;
}

int main(void) {
    double eps, d4, d5, i4, i5, n4, n5;
    d4 = dht(1.0, 3.0, *mainf4);
    d5 = dht(0.0, 1.0, *mainf5);
    i4 = iter(1.0, 3.0, *f4);
    i5 = iter(0.0, 1.0, *f5);
    n4 = newton(1.0, 3.0, *mainf4, *derivativef4);
    n5 = newton(0.0, 1.0, *mainf5, *derivativef5);
    eps = epsilon();
    printf("Машинное эпсилон: %.20f \n", eps);
    printf("| Выражение \t\t\t\t| Метод дихотомии \t| Метод итераций \t| Метод Ньютона \t|\n");
    printf("-----------------------------------------------------------------------------------------------------------------\n");
    printf("| 3 * x - 14 + exp(x) - exp(-x) \t| %.15lf \t| %.15lf \t| %.15lf \t|\n", d4, i4, n4);
    printf("| sqrt(1 - x) - tan(x) \t \t \t| %.15lf \t| %.15lf \t| %.15lf \t|\n", d5, i5, n5);
}