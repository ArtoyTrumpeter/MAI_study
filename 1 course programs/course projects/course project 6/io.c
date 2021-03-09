#include "io.h"

int readtxt(person *person, FILE *in)
{
    fscanf(in, "%s", person->surname);
    fscanf(in, "%s", person->initials);
    fscanf(in, " %c", &(person->gender));
    fscanf(in, "%d", &(person->school_number));
    fscanf(in, " %c", &(person->medal));
    fscanf(in, "%d", &(person->literature));
    fscanf(in, "%d", &(person->russian_language));
    fscanf(in, "%d", &(person->informatics));
    fscanf(in, "%d", &(person->physics));
    
    char fail_or_pass[4];
    fscanf(in, "%s", fail_or_pass);
    person->essay = strcmp(fail_or_pass, "pass");

    return !feof(in);
}

int readbin(person *person, FILE *in)
{
    fread(person->surname, sizeof(char), surname_LENGTH, in);
    fread(person->initials, sizeof(char), initials_LENGTH, in);
    fread(&(person->gender), sizeof(char), 1, in);
    fread(&(person->medal), sizeof(char), 1, in);
    fread(&(person->school_number), sizeof(int), 1, in);
    fread(&(person->literature), sizeof(int), 1, in);
    fread(&(person->russian_language), sizeof(int), 1, in);
    fread(&(person->informatics), sizeof(int), 1, in);
    fread(&(person->physics), sizeof(int), 1, in);
    fread(&(person->essay), sizeof(char), 1, in);
    return !feof(in);
}


void writebin(person *person, FILE *out)
{
    fwrite(person->surname, sizeof(char), surname_LENGTH, out);
    fwrite(person->initials, sizeof(char), initials_LENGTH, out);
    fwrite(&(person->gender), sizeof(char), 1, out);
    fwrite(&(person->medal), sizeof(char), 1, out);
    fwrite(&(person->school_number), sizeof(int), 1, out);
    fwrite(&(person->literature), sizeof(int), 1, out);
    fwrite(&(person->russian_language), sizeof(int), 1, out);
    fwrite(&(person->informatics), sizeof(int), 1, out);
    fwrite(&(person->physics), sizeof(int), 1, out);
    fwrite(&(person->essay), sizeof(char), 1, out);
}

void print(person *person) {
    printf(
        "Фамилия: %s Инициалы: %s Пол: %c. Есть медаль: %c ",
        person->surname,
        person->initials,
        person->gender,
        person->medal
    );
    printf(
        "Все оценки: %d %d %d %d ",
        person->literature,
        person->russian_language,
        person->informatics,
        person->physics
    );
    if (person->essay == 0) {
        printf("Эссе сдал.");
    } else {
        printf("Эссе не сдал.");
    }
    printf("\n");
}