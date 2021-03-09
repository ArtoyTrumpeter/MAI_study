#include "my_data.h"
#define key_length 6

void help() {
    printf("Standart add: press f\n");
    printf("Reverse add: press b\n");
    printf("Random add: press r\n");
}

int main(int argc, char *argv[]) 
{   
    if (argc != 2) {
        printf("Usage:./generate file.txt\n");
        exit(0);
    }
    table my_table;
    table_init(&my_table);
    char c;
    int a;
    int size, count1 = 0;
    FILE *in = fopen(argv[1], "w");
        if (!in) {
            printf("Error\n");
            exit(1);
        }
    help();
    scanf("%c", &c);
        switch(c) {
            case 'f':
            printf("Your size\n");
            scanf("%d", &size);
            for (int i = 0; i < size; i++) {
                for(char key1 = 'a'; key1 < 'z'; key1++) {
                    for(char key2 = 'a'; key2 < 'z'; key2++) {
                        for(char key3 = 'a'; key3 < 'z'; key3++) {
                            for(char key4 = 'a'; key4 < 'z'; key4++) {
                                for(char key5 = 'a'; key5 < 'z'; key5++) {
                                    for(char key6 = 'a'; key6 < 'z'; key6++) {
                                        if (size == count1) {
                                            break;
                                        }
                                        fprintf(in,"%c%c%c%c%c%c lol", key1, key2, key3, key4, key5, key6);
                                        fprintf(in,"\n");
                                        count1++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            break;
            case 'b':
            printf("Your size\n");
            scanf("%d", &size);
            for (int i = 0; i < size; i++) {
                for(char key1 = 'z'; key1 > 'a'; key1--) {
                    for(char key2 = 'z'; key2 > 'a'; key2--) {
                        for(char key3 = 'z'; key3 > 'a'; key3--) {
                            for(char key4 = 'z'; key4 > 'a'; key4--) {
                                for(char key5 = 'z'; key5 > 'a'; key5--) {
                                    for(char key6 = 'z'; key6 > 'a'; key6--) {
                                        if (size == count1) {
                                            break;
                                        }
                                        fprintf(in,"%c%c%c%c%c%c lol", key1, key2, key3, key4, key5, key6);
                                        fprintf(in,"\n");
                                        count1++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            break;
            case 'r':
            printf("Your size\n");
            scanf("%d", &size);
            for(int i = 0; i < size ; i++) {
                srand(i);
                for (int j = 0; j < key_length; j++) {
                    fprintf(in, "%c", (char)('!' + rand() % 93));
                }
                fprintf(in," lol\n");
            }
            break;
        }
    fclose(in);
    return 0;
}