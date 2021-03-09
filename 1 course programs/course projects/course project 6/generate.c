#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io.h"
#include "person.h"


int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage: ./generate <txt_file_from> <txt_file_to>\n");
        exit(0);
    }

    FILE *in = fopen(argv[1], "r");
    FILE *out = fopen(argv[2], "w");

    if (!in || !out) {
        printf("Error: could not open file\n");
        exit(1);
    }

    person current_person;

    while (!feof(in)) {
        readtxt(&current_person, in);
        writebin(&current_person, out);
    }

    fclose(in);
    fclose(out);
    return 0;
}