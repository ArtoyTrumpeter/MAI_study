#include "BST.h"

node* create (node *root, int info) { //инициализируем корень древа
    node *temp = (node*) malloc(sizeof(node)); //выделяем память
    temp->info = info; // присваиваем значение корню
    temp->parent = NULL; // обнуляем родителя
    temp->right = NULL; // обнуляем детей
    temp->left = NULL;
    root = temp;
    return root ;
}

node* add (node *root, int info) {
    node *root2 = root;
    node *root3 = NULL; // не забудем адрес родителя
    node *temp = (node*) malloc(sizeof(node)); // выделим память
    temp->info = info; // присваиваем значение
    while (root2 != NULL) { // ищем позицию для вставки
        root3 = root2;  
        if (info < root2->info) {
            root2 = root2->left;
        }
        else {
            root2 = root2->right;
        }
    }
    temp->parent = root3; // присвоили указатель на родителя, который нашли выше
    temp->left = NULL; //обнуляем детин
    temp->right = NULL;
    if (info < root3->info) { //вставляем узел в найденное место
        root3->left = temp;
    }
    else {
        root3->right = temp;
    }
    return root;
}

node* search (node *root, int info) { //поиск элэмента
    if ((root == NULL) || (root->info == info)) {
        return root; //если дерево пусто, либо его данные равны искомым, возвращаем указатель на дерево
    }
    if (info < root->info) {
        return search(root->left, info);
    }
    else {
        return search(root->right, info);
    }
}

node* min (node *root) { // поиск минимального элемента
    node *min = root;
    while ((min != NULL) && (min->left != NULL)) {
        min = min->left;
    }
    return min;
}

node* max (node *root) { // поиск максимального элемента 
    node *max = root;
    while((max != NULL) && (max->right != NULL)) {
        max = max->right;
    }
    return max;
}

node* following_node (node *root) { // поиск следующего по значению за данным элементом элемента
    node* p = root;
    node *searching_el = NULL;
    if (p->right != NULL) {
        return(min(p->right));
    }
    searching_el = p->parent;
    while ((searching_el != NULL) && (p ==  searching_el->right)) {
        p = searching_el;
        searching_el = searching_el->parent;
    }
    return searching_el;
}

node* delete (node *root, int info) {
    node* deleted_el = search(root, info);
    node* m = NULL;
    if ((deleted_el->left == NULL) && (deleted_el->right == NULL)) { //leave
        m = deleted_el->parent;
        if (deleted_el == m->right) {
            m->right = NULL;
        } else {
            m->left = NULL;
        }
        free(deleted_el);
    } else if ((deleted_el->left != NULL) && (deleted_el->right == NULL)){ //there is only left son
        m = deleted_el->parent;
        if (deleted_el == m->right) {
            m->right = deleted_el->left;
        } else {
            m->left = deleted_el->left;
        }
        free(deleted_el);
    } else if ((deleted_el->left == NULL) && (deleted_el->right != NULL)) { // there is only right son
        m = deleted_el->parent;
        if (deleted_el == m->right) {
            m->right = deleted_el->right;
        } else {
            m->left = deleted_el->right;
        }
        free(deleted_el);
    } else { // there are 2 sons
        m = following_node(deleted_el);
        deleted_el->info = m->info;
        if (m->right == NULL) {
            m->parent->left = NULL;
        } else {
            m->parent->left = m->right;
        }
        free(m);
    }
    return root;
}

void print (node *root) { // печать древа в прямом порядке
    if (root == NULL) {
        printf("NULL \n");
        return;
    }
    else {
        printf("%d \n", root->info);
    }
    print(root->left);
    print(root->right);
}

void tree_delete(node *root) {
    if (root) {
        tree_delete(root->left);
        tree_delete(root->right);
        root->left = NULL;
        root->right = NULL;
        root->parent = NULL;
        free(root);
    }
}