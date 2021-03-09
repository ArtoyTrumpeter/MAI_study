#ifndef IO_H_
#define IO_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "person.h"

int readtxt(person *person, FILE *in);
int readbin(person *person, FILE *in);
void writebin(person *person, FILE *out);
void print(person *person);

#endif