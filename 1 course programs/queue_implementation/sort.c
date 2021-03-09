#include <stdio.h>
#include <stdlib.h>
#include "sort.h"
#include "queue.h"

void sort_step(queue* root, int index) {
    if (root->data[index] >= root->data[index + 1]) {
        int a = root->data[index];
        int b = root->data[index + 1];
        for (int i = 0; i <=root->size - 1; i++) {
            if (i == index) {
                queue_push(root, b);
                queue_pop(root);
            }
            else if (i == index + 1) {
                queue_push(root, a);
                queue_pop(root);
            } else {
                queue_push(root, root->data[0]);
                queue_pop(root);
            }
        }
    }
}

void queue_sort(queue* root) {
    for (int i = 0; i <= root->size - 2; i++) {
        for (int j = 0; j <= root->size - i - 2; j++) {
            sort_step(root, j);
        }
    }
}