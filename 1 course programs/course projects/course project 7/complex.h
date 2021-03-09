#ifndef COMPLEX_H_
#define COMPLEX_H_
#include "stdio.h"
#define compl 2
#include "stdbool.h"
#include "stdlib.h"

typedef struct {
    double re;
    double im;
} complex;

typedef struct {
    complex *YE;
    int *PI;
    int *CIP;
    int ind;
    int sizeCIP;
    int size;
    int cips;
} vector;

void check_count_of_columns_and_lines(int fl,int fc);// проверяет равно ли количество строк количеству столбцов
void addCIP(vector *vect,int k);
void addPI(vector *v,int j);
void sum_vector(vector *v, vector *c,vector *s); // сумма
double addYE(vector *v,double c, double b);
void init_vector(vector *v);
void destroy_vector(vector *v);
void print_matrix(vector *v, int columns,int line);
void print_vector(vector *v);
void mult(vector *v, int a);
// за все остальное говорят названия
#endif 