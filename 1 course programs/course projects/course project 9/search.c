#include "my_data.h"
#include <string.h>

void help() {
    printf("Found element: press f\n");
    printf("Exit: press e\n");
    printf("Help: press h\n");
}


double binary_search(table *table, const char* key, int left, int right) {
    int mid = left + (right - left) / 2;
    if (strcmp(table->dt[mid].key, key) == 0) {
        printf("key:%s value:%s position:%d\n", table->dt[mid].key,
        table->dt[mid].info, mid + 1);
        return mid;
    } else if (left >= right) {
        printf("Table dosen't have elements with this key;\n");
        return -1;
    } else if (strcmp(table->dt[mid].key, key) > 0) {
        return binary_search(table, key, left, mid);
    } else if (strcmp(table->dt[mid].key, key) < 0) {
        return binary_search(table, key, mid + 1, right);
    }
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage:./search sort_txt_file \n");
        exit(0);
    }
    table my_table;
    char keys[7];
    table_init(&my_table);
    FILE *in = fopen(argv[1], "r");
        if (!in) {
            table_destroy(&my_table);
            printf("Error\n");
            exit(1);
        }
    table_txt_file(&my_table, in);
    fclose(in);
    char c;
    help();
    while (1) {
        scanf("%c", &c);
        switch(c) {
            case 'f':
            printf("Your key is:\n");
            scanf("%s", keys);
            binary_search(&my_table, keys, 0, (&my_table)->size);
            break;
            case 'e':
            table_destroy(&my_table);
            return 0;
            break;
            case 'h':
            help();
            break;
        }
    }
    return 0;
}