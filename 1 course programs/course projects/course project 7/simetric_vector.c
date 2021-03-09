#include "complex.h"


void print_vector(vector *v) {
    printf("CIP ");
    for(int i = 0; i < v->sizeCIP;i++) {
        printf("%d; ", v->CIP[i]);
    }
    printf("\n");
    if (v->size >= 1) {
        printf("YE ");
    }
    for(int i = 0; i < v->size;i++) {
        printf("%.1lf %.1lf; ", v->YE[i].re, v->YE[i].im);
    }
    if (v->size >= 1) {
        printf("\n");
        printf("PI ");
    }
    for(int i = 0; i < v->size;i++) {
        printf("%d; ", v->PI[i]);
    }
    if (v->size >= 1) {
        printf("\n");
    }
    
}


void init_vector(vector *v) {
    v->size = 0;
    v->sizeCIP = 0;
    v->ind = -1;
    v->CIP = (int *)malloc(sizeof(int));
    if (v->CIP == NULL) {
		exit(1);
	}
    v->PI = (int *)malloc(sizeof(int));
    if (v->PI == NULL) {
		exit(1);
	}
    v->YE = (complex *)malloc(sizeof(complex));
    if (v->YE == NULL) {
		exit(1);
	}
}


void addCIP(vector *v,int k) {
        v->sizeCIP++;
        v->CIP = (int *)realloc(v->CIP, sizeof(int) * (v->sizeCIP));
        v->CIP[v->sizeCIP - 1] = k;
}

void addPI(vector *v,int j) {
        v->PI = (int *)realloc(v->PI, sizeof(int) * (v->size));
        v->PI[v->size - 1] = j; 
    
}


double addYE(vector *v,double c, double d) {
        v->YE = (complex *)realloc(v->YE, sizeof(complex) * (compl * (v->size)));
        (v->YE[v->size - 1]).re = c;
        (v->YE[v->size - 1]).im = d;
    return 0;
}


void destroy_vector(vector *v) {
    free(v->PI);
    free(v->CIP);
    free(v->YE);
}