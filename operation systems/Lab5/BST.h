#ifndef _BST_
#define _BST_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <dlfcn.h>

typedef struct tree {
    int info;
    struct tree *parent;
    struct tree *left;
    struct tree *right;
} node;

node* create(node *root, int info);
node* add(node *root, int info);
node* search(node *root, int info);
node* min(node *root);
node* max(node *root);
node* folowing_node(node* root);
node* delete(node *root, int info);
void print(node *root);
void tree_delete(node *root);

#endif //_BST_