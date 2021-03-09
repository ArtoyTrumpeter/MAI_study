#include <stdio.h>
#include <stdlib.h>
#include "queue.h"

void queue_init(queue* root, int info) {
    root->data = (data_type*) malloc(sizeof(data_type) * 1);
    if (root->data == NULL) {
        exit(1);
    }
    root->size = 1;
    root->data[root->size - 1] = info;
}


bool is_queue_empty(queue* root) {
    return root->size == 0;
}

void queue_pop(queue* root) {
    for (int i = 0; i <= root->size - 1; i++) {
        root->data[i] = root->data[i + 1];
    }
    --root->size;
    root->data = (data_type*)realloc(root->data, sizeof(data_type) * (root->size));
}

int queue_size(queue* root) {
    return root->size;
}

int queue_push(queue* root, int info) {
    ++root->size;
    root->data = (data_type*)realloc(root->data, sizeof(data_type) * (root->size));
    root->data[root->size - 1] = info;
}

void queue_print(queue* root) {
    for (int i = 0; i <= root->size - 1; i++) {
        printf("%d ", root->data[i]);
    }
    printf("\n");
}

void queue_destroy(queue* root) {
    free(root->data);
    root->size = 0;
}