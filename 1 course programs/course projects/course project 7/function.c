#include "complex.h"

void check_count_of_columns_and_lines(int fl,int fc) {
    if (fl != fc || fl == 0 || fc == 0) {
        printf("ERROR\n");
        exit(2);
    }
}


double sum_of_re(vector *v, vector *c, int a,int b) {
    return (v->YE[a].re + c->YE[b].im);
}


double sum_of_im(vector *v,vector *c, int a,int b) {
    return (v->YE[a].im + c->YE[b].im);
}

void mult(vector *v, int a) {
    for(int i = 0; i < v->size; i++) {
       v->YE[i].re = a * v->YE[i].re;
       v->YE[i].im = a * v->YE[i].im;
    }
}

int v_cip(vector *v,int c) {
    c++;
    while((v->CIP[c] == 0) && (c < v->sizeCIP)) {
        c++;
    }
    if(v->CIP[c] != 0) {
        return v->CIP[c];
    } else {
        return v->ind;
    }
}

// сложение когда обе строки не пустые
void sum_of_two_vectors(vector *v,vector *c,vector *s,int ogr_f,int ogr_s,int a) { 
    int nv = v->CIP[a] - 1;
    int mc = c->CIP[a] - 1;
    int r = 1;
    while((nv < ogr_f) && (mc < ogr_s)) {
        if(v->PI[nv] > c->PI[mc]) {
        s->size++;
        addPI(s,c->PI[mc]);
        addYE(s,c->YE[mc].re,c->YE[mc].im);
        if(r == 1) {
            addCIP(s,s->size);
            r = 0;
            }
        mc++;  
        } else if(v->PI[nv] < c->PI[mc]) {
            s->size++;
            addPI(s,v->PI[nv]);
            addYE(s,v->YE[nv].re,v->YE[nv].im);
                if(r == 1) {
                    addCIP(s,s->size);
                    r = 0;
                }
                nv++;
        } else if(v->PI[nv] == c->PI[mc]) {
            if((sum_of_re(v,c,nv,mc) != 0) || (sum_of_im(v,c,nv,mc) != 0)) {
                s->size++;
                addPI(s,v->PI[nv]);
                addYE(s,sum_of_re(v,c,nv,mc),sum_of_im(v,c,nv,mc));
                if(r == 1) {
                    addCIP(s,s->size);
                    r = 0;
                }
            }
                mc++;
                nv++;
        }
    }
    if((nv == ogr_f) && (mc < ogr_s)) {
        for (int b = mc; b < ogr_s;b++) {
            s->size++;
            addYE(s,c->YE[b].re,c->YE[b].im);
            if(r == 1) {
                addCIP(s,s->size);
                r = 0;
            }
            addPI(s,c->PI[b]);  
        }
    } else if((mc == ogr_s) && (nv < ogr_f)) {
        for(int b = nv;b < ogr_f;b++) {
        s->size++;
        addYE(s,v->YE[b].re,v->YE[b].im);
            if(r == 1) {
                addCIP(s,s->size);
                r = 0;
            }
        addPI(s,v->PI[b]);     
        }
    }
    if(r == 1) {
        addCIP(s,0);
    }
}


void print_line(vector *v,int columns,int line,int limit) {
    int amount = 0;
    int b = v->CIP[line] - 1;
    while(amount < columns) {
        if((amount < v->PI[b]) && (b < limit)) {
        amount++;
        printf("0 + 0i ");
        } else if((amount > v->PI[b]) && (b < limit)){
            b++;
            printf("0 ");
        } else if((amount == v->PI[b]) && (b < limit)) {
            printf("%.0lf + %.0lfi ", v->YE[b].re, v->YE[b].im);
            amount++;
            b++;
        } else {
            amount++;
            printf("0 + 0i ");
        }
    }
}


void print_matrix(vector *v, int columns,int line) {
    int a = 0;
    if(v->sizeCIP != 1) {
        while(a < v->sizeCIP - 1) {
            if(v->CIP[a] == 0) {
                for(int i = 0;i < columns;i++) {
                    printf("0 + 0i ");
                }
            } else if(v->CIP[a] != 0) {
                if(v->CIP[a + 1] - v->CIP[a] > 0) {
                   print_line(v,columns,a,v->CIP[a + 1] - 1);
                } else if(v->CIP[a + 1] - v->CIP[a] < 0) {
                    if(v_cip(v,a) == v->ind) {
                      print_line(v,columns,a,v->size);  
                    } else if(v_cip(v,a) - v->CIP[a] > 0) {
                        int temp = v_cip(v,a) - 1;
                        print_line(v,columns,a,temp);
                    }
                }
            }
            a++;
            printf("\n");
        }
        if(v->CIP[a] == 0) {
            for(int i = 0;i < columns;i++) {
                printf("0 + 0i ");
            }
        } else {
            print_line(v,columns,a,v->size);
        }
        printf("\n");
    } else {
        print_line(v,columns,a,v->size);
    }
}


void sum_of_one_vector(vector *c,vector *s,int ogr,int a) {
    int r = 1;
    for(int b = c->CIP[a] - 1;b < ogr;b++) {
        s->size++;
        addYE(s,c->YE[b].re,c->YE[b].im);
        if(r == 1) {
            addCIP(s,s->size);
            r = 0;
        }
        addPI(s,c->PI[b]);
    }   
}

void sum_vector(vector *v, vector *c,vector *s) {
    int a = 0;
    int r = 1;
    if(v->sizeCIP != 1) {
        while(a < v->sizeCIP - 1) {
            if(v->CIP[a] == 0 || c->CIP[a] == 0) {
                if(v->CIP[a] == 0 && c->CIP[a] == 0) {//ok
                    if(r == 1) {
                        addCIP(s,0);
                        r = 0;
                    } 
                } else if((v->CIP[a] == 0) && (c->CIP[a] != 0)) {//ok
                    if(c->CIP[a + 1] - c->CIP[a] > 0) {
                        sum_of_one_vector(c,s,c->CIP[a + 1] - 1,a);  
                    } else if(c->CIP[a + 1] - c->CIP[a] < 0) {
                        if(v_cip(c,a) - c->CIP[a] > 0) {
                            int temp = v_cip(c,a) - 1;
                            sum_of_one_vector(c,s,temp,a);
                        } else if(v_cip(c,a) == c->ind) {
                            sum_of_one_vector(c,s,c->size,a);
                        }
                    } 
                } else if((v->CIP[a] != 0) && (c->CIP[a] == 0)) {//ok
                    if(v->CIP[a + 1] - v->CIP[a] > 0) {
                        sum_of_one_vector(v,s,v->CIP[a + 1] - 1,a);
                    } else if(v->CIP[a + 1] - v->CIP[a] < 0) {
                        if(v_cip(v,a) - v->CIP[a] > 0) {
                            int temp = v_cip(v,a) - 1;
                            sum_of_one_vector(v,s,temp,a);
                        } else if(v_cip(v,a) == v->ind) {
                            sum_of_one_vector(v,s,v->size,a);
                        } 
                    }
                }
            } else {
                if((v->CIP[a+1] - v->CIP[a] > 0) && (c->CIP[a+1] - c->CIP[a] > 0)) {//ok2
                    sum_of_two_vectors(v,c,s,v->CIP[a+1] - 1,c->CIP[a+1] - 1,a);
                } else if((v->CIP[a+1] - v->CIP[a] < 0) && (c->CIP[a+1] - c->CIP[a] < 0)) {//ok3
                    if ((v_cip(v,a) - v->CIP[a] > 0) && (v_cip(c,a) - c->CIP[a] > 0)) {
                        int temp_v = v_cip(v,a) - 1;
                        int temp_c = v_cip(c,a) - 1;
                        sum_of_two_vectors(v,c,s,temp_v,temp_c,a);
                    } else if ((v_cip(v,a) - v->CIP[a] > 0) && (v_cip(c,a) == c->ind)) {
                        int temp = v_cip(v,a) - 1;
                        sum_of_two_vectors(v,c,s,temp,c->size,a);
                    } else if ((v_cip(v,a) == v->ind) && (v_cip(c,a) - c->CIP[a] > 0)) {
                        int temp = v_cip(c,a) - 1;
                        sum_of_two_vectors(v,c,s,v->size,temp,a);
                    } else if((v_cip(v,a) == v->ind) && (v_cip(c,a) == c->ind)) {
                        sum_of_two_vectors(v,c,s,v->size,c->size,a);
                    }
                } else if((v->CIP[a+1] - v->CIP[a] < 0) && (c->CIP[a+1] - c->CIP[a] > 0)) {//ok4
                    if(v_cip(v,a) - v->CIP[a] > 0) {
                        int temp = v_cip(v,a) - 1;
                        sum_of_two_vectors(v,c,s,temp,c->CIP[a+1] - 1,a);
                    } else if(v_cip(v,a) == v->ind) {
                        sum_of_two_vectors(v,c,s,v->size,c->CIP[a+1] - 1,a);
                    }
                } else if((v->CIP[a+1] - v->CIP[a] > 0) && (c->CIP[a+1] - c->CIP[a] < 0)) {//ok5
                     if(v_cip(c,a) - c->CIP[a] > 0) {
                         int temp = v_cip(c,a) - 1;
                        sum_of_two_vectors(v,c,s,v->CIP[a + 1] - 1,temp,a);
                    } else if(v_cip(c,a) == c->ind) {
                        sum_of_two_vectors(v,c,s,v->CIP[a + 1] - 1,c->size,a);
                    } 
                }
            }
            a++;
            r = 1;
        }
        if ((v->CIP[a] == 0) && (c->CIP[a] == 0)) {//ok
            if(r == 1) {
                addCIP(s,0);
                r = 0;
            }
        } else if((v->size - (v->CIP[a] - 1) > 0) && (c->CIP[a] == 0)) {//ok
            sum_of_one_vector(v,s,v->size,a); 
        } else if((v->CIP[a] == 0) && (c->size - (c->CIP[a] - 1) > 0)) { //ok
            sum_of_one_vector(c,s,c->size,a);
        } else if ((v->size - (v->CIP[a]- 1) > 0) && (c->size - (c->CIP[a] - 1)  > 0)) {//ok
            sum_of_two_vectors(v,c,s,v->size,c->size,a);
        }    
    } else {//ok
         sum_of_two_vectors(v,c,s,v->size,c->size,a);
    }
}