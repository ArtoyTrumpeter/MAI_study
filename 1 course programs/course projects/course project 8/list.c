#include "list.h"

void list_init(list **root, int info) {
    *root = (list*) malloc (sizeof(list));
    (*root)->info = info;
    (*root)->next = *root;
}

void list_push(list **root, int info) {
    list *temp_add, *temp_go_to_end;
    temp_add = (list*) malloc(sizeof(list));
    temp_go_to_end = *root;
    while(temp_go_to_end->next != *root) {
        temp_go_to_end = temp_go_to_end->next;
    }
    temp_go_to_end->next = temp_add;
    temp_add->info = info;
    temp_add->next = *root;
}

void list_pop(list **root) {
    list *temp_go_to_end, *temp_popping;
    temp_go_to_end = *root;
    while(temp_go_to_end->next->next != *root) {
        temp_go_to_end = temp_go_to_end->next;
    }
    temp_popping = temp_go_to_end->next;
    temp_go_to_end->next = *root;
    if (*root == temp_popping) {
        *root = NULL;
    }
    free(temp_popping);
}

void list_print(list *root) {
    if (root == NULL) {
        printf("\n");
        return;
    } else {
        printf("%d ", root->info);
        list *temp_go_to_end;
        temp_go_to_end = root;
        while(temp_go_to_end->next != root) {
            temp_go_to_end = temp_go_to_end->next;
            printf("%d ", temp_go_to_end->info);
        }
    }
}

int list_size(list *root) {
    if (root == NULL) {
        return 0;
    } else {
        int size = 1;
        list *temp_go_to_end;
        temp_go_to_end = root;
        while(temp_go_to_end->next != root) {
            temp_go_to_end = temp_go_to_end->next;
            size++;
        }
        return size;
    }
}

void list_delete_k_elements(list **root, int count_of_elements) {
if (list_size(*root) < count_of_elements) {
        return;
    }
    for (int i = 0; i < count_of_elements; i++) {
        list_pop(root);
    }
}