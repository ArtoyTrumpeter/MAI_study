#ifndef person_h_
#define person_h_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define surname_LENGTH 32
#define initials_LENGTH 5

typedef struct {
	char surname[surname_LENGTH];
	char initials[initials_LENGTH];
	char gender;
	char medal;
    int school_number;
	int literature;
	int russian_language;
	int informatics;
	int physics;
	char essay;
} person;
#endif