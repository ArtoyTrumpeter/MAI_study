#include <stdio.h>
#include <stdlib.h>
#include "tree.h"

tree* tree_init (tree** root, int info) { //инициализируем корень древа
    tree *temp = (tree*)(malloc(sizeof(tree))); //выделяем памяттттт
    temp->info = info; // присваиваем значение корню
    temp->parent = NULL; // обнуляем родителя
    temp->right = NULL; // обнуляем детину правую
    temp->left = NULL; // обнуляем детину левую
    *root = temp;
    return temp;
}

tree* branch_add (tree *root, int info) { // добавим детин
    tree *parent = root; // не забудем адрес родителя.....
    tree *temp = (tree*)(malloc(sizeof(tree))); // выделим память
    temp->info = info; // присваиваем значение
    while(parent != NULL) { // ищем позицию для вставки
        if(info < parent->info) {
            if (!parent->left) {
                break;
            }
            parent = parent->left;
        }
        else {
            if (!parent->right) {
                break;
            }
            parent = parent->right;
        }
    }
    temp->parent = parent; // присвоили указатель на родителя, который нашли выше
    temp->left = NULL; //обнуляем детин
    temp->right = NULL;
    if(info < parent->info) { //вставляем узел в найденное место
        parent->left = temp;
    }
    else {
        parent->right = temp;
    }
    return root;
}

tree* branch_search (tree *root, int info) { //поиск элэмента
    if((root == NULL) || (root->info == info)) {
        return root; //если дерево пусто, либо его данные равны искомым, возвращаем указатель на дерево
    }
    if(info < root->info) {
        return branch_search(root->left, info);
    }
    else {
        return branch_search(root->right, info);
    }
}

tree* branch_min (tree *root) { // поиск минимального элемента
    tree *min = root;
    while((min != NULL) && (min->left != NULL)) {
        min = min->left;
    }
    return min;
}

tree* branch_max (tree *root) { // поиск максимального элемента 
    tree *max = root;
    while((max != NULL) && (max->right != NULL)) {
        max = max->right;
    }
    return max;
}

tree* branch_find_following (tree *root) { // поиск следующего по значению за данным элементом элемента
    return root->right ? branch_min(root->right) : branch_max(root->left);
}

void branch_delete(tree** root, int info) {
    tree* deleted_node = branch_search(*root, info);
    tree* replacing_node = branch_find_following(deleted_node);
    
    if (replacing_node == NULL) { // удаляемый узел - лист
        if (deleted_node->parent != NULL) {
            if (deleted_node->parent->right == deleted_node) {
                deleted_node->parent->right = NULL;
            } else {
                deleted_node->parent->left = NULL;
            }

        } else { // удаляемый узел - корень
            *root = NULL;
        }

        free(deleted_node);
        
    } else { 
        deleted_node->info = replacing_node->info; 

        if (replacing_node->parent->right == replacing_node) {
            replacing_node->parent->right = (
                replacing_node->right
                ? replacing_node->right
                : replacing_node->left
            );
        } else {
            replacing_node->parent->left = (
                replacing_node->right
                ? replacing_node->right
                : replacing_node->left
            );
        }

        free(replacing_node);

    }

}

void tree_print (tree *root, int level) { // печать древа
    for (int i = 0; i < level; i++) { printf(" "); }
    if(root == NULL) {
        printf("NULL \n");
        return;
    }
    else {
        printf("%d \n", root->info);
    }
    tree_print(root->left, level+1);
    tree_print(root->right, level+1);
}

int tree_count_leaves(tree* root) {
    if (!root) {
        return 0;
    }
    if (!(root->left) && !(root->right)) {
        return 1;
    }
    int count = 0;
    if (root->left) {
        count += tree_count_leaves(root->left);
    }
    if (root->right) {
        count += tree_count_leaves(root->right);
    }
    return count;
}