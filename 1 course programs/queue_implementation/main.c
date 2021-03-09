#include <stdio.h>
#include <stdlib.h>
#include "queue.h"
#include "sort.h"

int main() {
    int info, temp;
    printf("enter info of first elem: ");
    scanf("%d", &info);
    queue* root;
    queue_init(root, info);
    while(temp != 4) {
        printf("\n1 - push, 2 - pop, 3 - print queue, 4 - destroy, 5 - sort, 6 - size, 7 - exit\n");
        scanf("%d", &temp);
        if(temp == 1) {
            printf("enter element info: ");
            scanf("%d", &info);
            queue_push(root, info);
        }
        else if(temp == 2) {
            queue_pop(root);
        }
        else if(temp == 3) {
            queue_print(root);
        }
        else if(temp == 4) {
            queue_destroy(root);
        }
        else if(temp == 5) {
            queue_sort(root);
        }
        else if(temp == 6) {
            printf("%d\n", queue_size(root));
        }
        else {
            return 0;
        }
    }
}