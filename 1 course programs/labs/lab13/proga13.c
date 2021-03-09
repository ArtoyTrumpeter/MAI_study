#include <stdio.h>


int main(void) {
    int answr = 0, count = 0;
    char c;
    unsigned int letter_set = 0;
    while ((c = getchar()) != EOF ) {
        if ((c != 'a') && (c != 'e') && (c != 'y') && (c != 'u') && (c != 'i') && (c != 'o') && (c != ' ') && (c != '\n') && (c != '\t') && (c != ',')) {
            letter_set = letter_set | (1u << (c - 'a'));
            count++;
        }
        if ((c == ' ') || (c == '\n') || (c == '\t') || (c == ',')) {
            if (count == 1) {
                answr = 1;
            }
            letter_set = 0;
            count = 0;
        }
    }
    if (answr == 1) {
        printf("Yes\n");
    }
    else {
        printf("No\n");
    }
    return 0;
}
