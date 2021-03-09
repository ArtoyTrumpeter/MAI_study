#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "io.h"
#include "person.h"


int main(int argc, char *argv[])
{
	
	if (argc != 2) {
		printf("Usage: ./execute <bin_DB_file>\n");
		exit(0);
	}

	FILE *in = fopen(argv[1], "r");

	if (!in) {
		printf("Error: could not open file\n"); 
		exit(1);
	}

	person current_person;
	int avg = 0;
	int count = 0;
	while (readbin(&current_person, in)) {

		avg += current_person.literature;
		avg += current_person.russian_language;
		avg += current_person.informatics;
		avg += current_person.physics;
		count++;
	}
	avg /= count;

	fseek(in, 0, SEEK_SET);

	while(readbin(&current_person, in)) {
		if (
			(
				current_person.literature
				+ current_person.russian_language
				+ current_person.informatics
				+ current_person.physics
				> avg
			) && (
				current_person.medal == 'n'
			) && (
				current_person.gender == 'm'
			)
		) {
			print(&current_person);
		}
	}
    return 0;
}