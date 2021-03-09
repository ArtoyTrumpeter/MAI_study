#include "stdio.h"
#include "complex.h"
#include "stdlib.h"
#define compl 2


void help() {
    printf("Complex number: x + y*i;y != 0;\n");
    printf("At the terminal 3 + 4i = 3 4;\n");
}

int main(int argc, char *argv[])
{
    int r = 1;
    vector first;
    vector second;
    double real_part;
    double imaginary;
    int firlin, fircol;
    printf("The first matrix\n");
    printf("lines ");
    scanf("%d", &firlin);
    printf("columns ");
    scanf("%d", &fircol);
    check_count_of_columns_and_lines(firlin,fircol);
    init_vector(&first);
    help();
    for(int i = 0; i < firlin;i++) {
        for(int j = 0; j < fircol; j++) {
            scanf("%lf", &real_part);
            scanf("%lf", &imaginary);
            if (imaginary != 0 || real_part != 0) {
                (&first)->size++;
                (&first)->cips++;
                addYE(&first,real_part,imaginary);
                if(r == 1) {
                    addCIP(&first,(&first)->cips);
                    r = 0;
                }
                addPI(&first,j);
            }
        }
        if(r == 1) {
            addCIP(&first,0);
        }
        r = 1;
    }
    print_vector(&first);
    r = 1;
    init_vector(&second);
     help();
    for(int i = 0; i < firlin;i++) {
        (&second)->cips++;
        (&second)->size++;
        addYE(&second,1,0);
        addCIP(&second,r);
        addPI(&second,r - 1);
        r++;
    }
    print_vector(&second);
    vector third;
    init_vector(&third);
    printf("Your a and b?\n");
    int a,b;
    scanf("%d", &a);
    scanf("%d", &b);
    if(a == 0 && b == 0) {
        printf("Result matrix is zero matrix\n");
         destroy_vector(&first);
        destroy_vector(&second);
        destroy_vector(&third);
        return 0;
    } else if (a == 0) {
        mult(&second,b);
        print_matrix(&second,fircol,firlin);
        destroy_vector(&first);
        destroy_vector(&second);
        destroy_vector(&third);
        return 0;
    } else if (b == 0) {
        mult(&first,a);
        print_matrix(&first,fircol,firlin);
        destroy_vector(&first);
        destroy_vector(&second);
        destroy_vector(&third);
        return 0;
    }
    mult(&second,b);
    mult(&first,a);
    sum_vector(&first,&second,&third);
    printf("Result matrix\n");
    print_vector(&third);
    print_matrix(&third,fircol,firlin);
    destroy_vector(&first);
    destroy_vector(&second);
    destroy_vector(&third);
    return 0;
}