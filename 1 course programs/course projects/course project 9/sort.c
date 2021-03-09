#include "my_data.h"

void swap(table *my_table, int index) {
    char temp = *(my_table)->dt[index-1].key;
    *(my_table)->dt[index-1].key = *(my_table)->dt[index].key;
    *(my_table)->dt[index].key = temp;
}

void sort_of_simple_insert(table *my_table) {
    for(int i = 1; i < my_table->size; i++) {
        for(int j = i; (j > 0) && (my_table->dt[j-1].key > my_table->dt[j].key); j--) {
			    swap(my_table, j);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage:./sort no_sort_txt_file sort_txt_file\n");
        exit(0);
    }
    table my_table;
    table_init(&my_table);
    FILE *in = fopen(argv[1], "r");
        if (!in) {
            table_destroy(&my_table);
            printf("Error\n");
            exit(1);
        }
    table_txt_file(&my_table, in);
    fclose(in);
    sort_of_simple_insert(&my_table);
    int size_m = my_table.size;
    FILE *out = fopen(argv[2], "w");
        if (!out) {
            table_destroy(&my_table);
            printf("Error\n");
            exit(1);
        }
    for (int i = 0; i < size_m; i++) {
        fprintf(in,"%s %s ", my_table.dt[i].key, my_table.dt[i].info);
        fprintf(in,"\n");    
    }
    fclose(out); 
    table_destroy(&my_table);
    return 0;
}