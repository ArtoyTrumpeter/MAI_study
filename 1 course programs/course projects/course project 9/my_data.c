#include "my_data.h"

void table_init(table *table) {
    table->size = 0;
    table->dt = (data_type *)malloc(sizeof(data_type) * (type * table->size));
}

void table_destroy(table *table) {
        free(table->dt);
}

void table_txt_file(table *table, FILE *in) {
    int i = 0;
    int size_l = 20;
    table->dt = (data_type *)realloc(table->dt,sizeof(data_type) * (type * (size_l)));
    while(1) {
        if (!feof(in)) {
            if(i > size_l) {
                size_l = size_l * 2;
                table->dt = (data_type *)realloc(table->dt,sizeof(data_type) * (type * (size_l)));
            }
            table->dt[i].info = (char *)malloc(sizeof(char));
            fscanf(in,"%c", (table->dt[i].key));
            fscanf(in,"%c", table->dt[i].info);
            i++;
            table->size = i - 1;
        } else {
            break;
        }
    }
}