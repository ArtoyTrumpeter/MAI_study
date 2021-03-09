#include <stdio.h>
#include <stdlib.h>
//#include <pthread.h>

void bubble_sort (int* array, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = size - 1; j > i; j--) {
            if (array[j - 1] > array[j]) {
                int temp = array[j - 1];
                array[j - 1] = array[j];
                array[j] = temp;
            }
        }
    }
}

typedef struct {
    int size;
    int** table;
} Matrix;

int** mrx_table_memory_create(int size) { //выделение памяти под матрицу
    int** arr = (int**) malloc(sizeof(int*) * size);
    for (int i = 0; i < size; i++) {
        arr[i] = malloc(sizeof(int) * size);
    }
    return arr;
}

void mrx_full(Matrix* mrx) { //заполнение матрицы
    for (int i = 0; i < mrx->size; i++) {
        for (int j = 0; j < mrx->size; j++) {
            scanf("%d", &mrx->table[i][j]);
        }
    }
}

void window_full(Matrix* window, Matrix* mrx, int up_h, int up_l) { //заполнение окна
    int up_l_border = up_l;
    for (int i = 0; i < window->size; i++) {
        up_l = up_l_border;
        for (int j = 0; j < window->size; j++) {
            window->table[i][j] = mrx->table[up_h][up_l];
            up_l++;
            if (j == window->size - 1) {
                up_h++;
            }
        }
    }
}

Matrix* io_mrx_create(int size) { //создание пустой готовой матрицы
    Matrix* mrx = (Matrix*) malloc(sizeof(Matrix));
    mrx->table = mrx_table_memory_create(size);
    mrx->size = size;
    return mrx;
}

void mrx_print(Matrix* mrx) { //печать матрицы
    for (int i = 0; i < mrx->size; i++) {
        for (int j = 0; j < mrx->size; j++) {
            printf("%d", mrx->table[i][j]);
            printf(" ");
            if (j == mrx->size - 1) {
                printf("\n");
            }
        }
    }
}

int* mrx_to_array(Matrix* window) { //перевод матричного окна в массив
    int arr_size = window->size * window->size;
    int k = 0;
    int* arr = (int*) malloc(sizeof(int) * arr_size);
    for (int i = 0; i < window->size; i++) {
        for (int j = 0; j < window->size; j++) {
            arr[k] = window->table[i][j];
            k++;
        }
    }
    return arr;
}

int median_filter(Matrix* window) { //медианный фильтр
    int* arr = mrx_to_array(window);
    int arr_size = window->size * window->size;
    bubble_sort(arr, arr_size);
    int median_value = arr[arr_size / 2];
    return median_value;
}

void mrx_destroy(int** arr, int size) { //очистка памяти 
    for (int i = 0; i < size; i++) {
        free(arr[i]);
        arr[i] = NULL;
    }
    free(arr);
}

void io_mrx_destroy(Matrix* mrx) {
    mrx_destroy(mrx->table, mrx->size);
    mrx->table = NULL;
    free(mrx);
}

int main(int argc, char* argv[]) {
    /*if (argc != 2) {
        printf("Use like: ./a.out number_of_threads\n");
        exit(2);
    }*/
    //int num_threads = atoi(argv[1]);
    //pthread_t* threads = (pthread_t*) malloc(sizeof(pthread_t) * num_threads);
    int num_filtr;
    scanf("%d", &num_filtr);
    int win_size;
    scanf("%d", &win_size);
    while (win_size % 2 == 0) {
        printf("Enter odd size of window");
        printf("\n");
        scanf("%d", &win_size);
    }
    int size;
    scanf("%d", &size);
    Matrix* mrx = io_mrx_create(size);
    Matrix* window = io_mrx_create(win_size);
    mrx_full(mrx);
    while (num_filtr > 0) {
        int new_size = mrx->size - win_size + 1;
        Matrix* new_matrix = io_mrx_create(new_size);
        for (int i = 0; i < mrx->size - win_size + 1; i++) {
            for (int j = 0; j < mrx->size - win_size + 1; j++) {
                window_full(window, mrx, i, j);
                new_matrix->table[i][j] = median_filter(window);
            }
        } //осталось реализовать здесь потоки 
        mrx = (Matrix*) realloc(mrx, new_size);
        mrx->size = new_size;
        for (int i = 0; i < new_size; i++) {
            for (int j = 0; j < new_size; j++) {
                mrx->table[i][j] = new_matrix->table[i][j];
            }
        }
        io_mrx_destroy(new_matrix);     
        num_filtr--;
    }
    io_mrx_destroy(window);
    printf("\n");
    mrx_print(mrx);
    io_mrx_destroy(mrx);
    return 0;
}