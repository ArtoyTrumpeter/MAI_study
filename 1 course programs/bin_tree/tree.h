#ifndef TREE_H
#define TREE_H

typedef struct tree {
    int info;
    struct tree *left;
    struct tree *right;
    struct tree *parent;
} tree;

#endif