#include <stdio.h>
#include <stdlib.h>
#include "list.h"

void menu() {
    printf("0-menu, 1-push, 2-pop, 3-print, 4-size, 5-function, 6-exit\n");
}

int main() {
    int info, temp, count_of_elements;
    printf("enter info of first elem: ");
    scanf("%d", &info);
    list *root;
    list_init(&root, info);
    while(temp != 6) {
        printf("\nEnter your option: ");
        scanf("%d", &temp);
        printf("\n");
        switch (temp) {
            case 0:
                menu();
                break;
            case 1:
                printf("Enter the info of the node: ");
                scanf("%d", &info);
                if (list_size(root) > 0) {
                    list_push(&root, info);
                } else {
                    list_init(&root, info);
                }
                break;
            case 2:
                list_pop(&root);
                break;
            case 3:
                if (list_size(root) == 0) {
                    printf("The list is empty!\n");
                    break;
                }
                list_print(root);
                printf("\n");
                break;
            case 4:
                printf("Size of list = %d\n", list_size(root));
                break;
            case 5:
                printf("Enter count:");
                scanf("%d", &count_of_elements);
                list_delete_k_elements(&root, count_of_elements);
                break;
            case 6:
                printf("Exiting...\n");
                return 0;
                break;
        }
    }
    return 0;
}