#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <fcntl.h>
#include <string.h>


int main(int argc, char* argv[]) {
	char sym;
    int cap = 4;
	char* in = (char*) malloc(sizeof(char) * cap);
    char* out = argv[1];
    if (argc != 2) {
        perror("Use like: ./a.out <tofile>");
        exit(1);
    }
    int i = 0;
	while ((sym = getchar()) != EOF) {
        if (i > cap) {
            cap = cap * 4 / 3;
            in = (char*)realloc(in, sizeof(char) * cap);
        }
		in[i] = sym;
        i++;
	}
    in = (char*) realloc(in, sizeof(char) * i);
	execl("child", in, out, NULL); 
	return 0;
}