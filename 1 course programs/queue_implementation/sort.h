#ifndef _SORT_H_
#define _SORT_H_

#include <stdio.h>
#include "queue.h"

void queue_sort(queue* root);
void sort_step(queue* root, int index);

#endif //_SORT_H_