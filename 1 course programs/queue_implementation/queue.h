#ifndef QUEUE_H
#define QUEUE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef int data_type;

typedef struct queue {
    data_type* data;
    int size;
} queue;

void queue_init(queue* root, int info);
bool is_queue_empty(queue* root);
void queue_pop(queue* root);
int queue_size(queue* root);
int queue_push(queue* root, int info);
void queue_print(queue* root);
void queue_destroy(queue* root);


#endif //QUEUE_H