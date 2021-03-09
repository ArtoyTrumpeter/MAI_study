#ifndef _MY_TABLE_
#define _MY_TABLE_
#ifndef _NAME_
#define _NAME_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define type 3

typedef struct data_type {
    char *info;
    char *key;
} data_type;

typedef struct table {
    data_type *dt;
    int size;   
} table;

void table_init(table *table);
void table_destroy(table *table);
void table_txt_file(table *table, FILE *in);

#endif //_MY_TABLE_
#endif //_NAME_