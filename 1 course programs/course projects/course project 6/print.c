#include <stdio.h>
#include <stdlib.h>
# include <string.h>

#include "person.h"
#include "io.h"



int main(int argc, char *argv[])
{
	
	if (argc != 2) {
		printf("Usage: ./print <binary_DB_file>\n");
		exit(0);
	}

	FILE *in = fopen(argv[1], "r");
	
	if (!in) {
		printf("Error: could not open file\n"); 
		exit(1);
	}

	person s;

	while (readbin(&s, in)) {
		print(&s);
	}

	fclose(in);
	
	return 0;
}
