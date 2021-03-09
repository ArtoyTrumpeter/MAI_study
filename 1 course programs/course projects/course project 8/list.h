#ifndef _LIST_H_
#define _LIST_H_

#include <stdio.h>
#include <stdlib.h>

typedef struct list {
    struct list *next;
    int info;
} list;

void list_init(list **head, int info);
void list_push(list **head, int info);
void list_pop(list **head);
void list_print(list *head);
int list_size(list *head);
void list_delete_k_elements(list **head, int count_of_elemnts);

#endif //_LIST_H_