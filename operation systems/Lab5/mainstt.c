#include "BST.h"

void help() {
    printf("Add: press a \n");
    printf("Delete element: press d \n");
    printf("Print: press p \n");
    printf("Help: press h \n");
    printf("Exit: press e \n");
}

int main() {
    node *my_tree = NULL;
    int inf;
    char c;
    printf("Please, enter your first info ");
    scanf("%d", &inf);
    my_tree = create(my_tree, inf);
    help();
    while (true) {
        scanf("%c", &c);
        switch(c) {
            case 'a':
                printf("Enter your element ");
                scanf("%d", &inf);
                add(my_tree, inf);
            break;
            case 'd':
                printf("Enter your element ");
                scanf("%d", &inf);
                delete(my_tree, inf);
            break;
            case 'p':
                print(my_tree);
            break;
            case 'h':
                help();
            break;
            case 'e':
                tree_delete(my_tree);
                my_tree = NULL;
                return 0;
            break;
        }
    }
    return 0;
}