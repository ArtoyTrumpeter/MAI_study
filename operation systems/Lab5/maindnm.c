#include <dlfcn.h>
#include "BST.h"
#define Error_dlsym 5

void help() {
    printf("Add: press a \n");
    printf("Delete element: press d \n");
    printf("Print: press p \n");
    printf("Help: press h \n");
    printf("Exit: press e \n");
}

void* get(void *libHandle, char* name) {
    void* temp = dlsym(libHandle, name);
    if (temp == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        exit(Error_dlsym);
    }
    return temp;
}

int main() {
    void *libHandle;
    libHandle = dlopen("./libBST.so", RTLD_LAZY);
    if (!libHandle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(1);
    }
    node*(*create)(node *my_tree) = get(libHandle, "create");
    node*(*add)(node *my_tree, int inf) = get(libHandle, "add");
    node*(*delete)(node *my_tree, int inf) = get(libHandle, "delete");
    void(*print)(node *my_tree) = get(libHandle, "print");
    void(*tree_delete)(node *my_tree) = get(libHandle, "tree_delete");
    node *my_tree = NULL;
    int inf;
    char c;
    printf("Please, enter your firt info ");
    scanf("%d", &inf);
    my_tree = (*create)(my_tree);
    help();
    while (true) {
        scanf("%c", &c);
        switch(c) {
            case 'a':
                printf("Enter your element ");
                scanf("%d", &inf);
                (*add)(my_tree, inf);
            break;
            case 'd':
                printf("Enter your element ");
                scanf("%d", &inf);
                (*delete)(my_tree, inf);
            break;
            case 'p':
                (*print)(my_tree);
            break;
            case 'h':
                help();
            break;
            case 'e':
                (*tree_delete)(my_tree);
                my_tree = NULL;
                dlclose(libHandle);
                return 0;
            break;
        }
    }
    return 0;
}