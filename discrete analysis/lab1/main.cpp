#include <iostream>

using namespace std;

struct elements {
    char* info;
    int key;
};

int maximum(elements *table, int size) {
    int max = 0;
    for (int i = 0; i < size; i++) {
        if ((table[i].key) > max) {
            max = table[i].key;
        }
    }
    return max;
}

elements* counting_sort(elements * table, int size, int max) {
    elements *sorted_table = new elements [size];
    int *count_table = new int [max + 1];
    for (int i = 0 ; i < (max + 1); i++) {
        count_table[i] = 0;
    }
    for (int i = 0 ; i < size; i++) {
        count_table[table[i].key]++;
    }
    for (int i = 1 ; i < (max + 1); i++) {
        count_table[i] = count_table[i] + count_table[i - 1];
    }
    for (int i = (size - 1); i >= 0; i--) {
        sorted_table[--count_table[table[i].key]].key = table[i].key;
        sorted_table[count_table[table[i].key]].info = table[i].info;
    }
    delete [] count_table;
    delete [] table;
    return sorted_table;
}

elements* resize(elements* table, int size) {
    elements* temp = new elements [size * 2];
    for (int i = 0; i < size; i++) {
        temp[i] = table[i];
    }
    delete [] table;
    return temp;
}

int main() {
    ios_base::sync_with_stdio(false);
    int size = 0;
    int iterator = 0;
    int max_size = 2;
    elements *table = new elements [2];
    table[iterator].info = new char[65];
    while ((scanf("%d %s", &table[iterator].key, table[iterator].info)) == 2) {
        iterator++;
        size = iterator;
        if (size == max_size) {
            table = resize(table, size);
            max_size *= 2;
        }
        table[iterator].info = new char[65];
    }
    delete [] table[iterator].info;
    table = counting_sort(table, size, maximum(table, size));
    for (int i = 0; i < size; i++) {
        cout << table[i].key << " " << table[i].info << "\n";
        delete [] table[i].info;
    }
    delete [] table;
    return 0;
}