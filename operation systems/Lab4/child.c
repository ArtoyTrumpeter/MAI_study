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
    int fdout;
    char c;
    void *dst;
    int cap = 4;
	char* out = (char*) malloc(sizeof(char) * cap);
    char* in = argv[0];
    char* output = argv[1];
    int count = 0;
	bool lspace = false;
    for (int i = 0; (c = in[i]) != '\0'; ++i) {
        //printf("%d\n", i);
        if (c == ' ' && lspace) continue;
        if (i > cap) {
            cap = cap * 4 / 3;
            out = (char*)realloc(out, sizeof(char) * cap);
        }
        out[count] = c;
        count++;
    	if (c == ' ') {
    	    lspace = true;
        } else {
    		lspace = false;
    	}
    }
    out[count] = '\0';
    if ((fdout = open(output, O_CREAT | O_RDWR | O_TRUNC, 0666)) < 0 ){
        perror("Невозможно создать файл для записи");
        exit(1);
    }
    /*установить размер выходного файла*/
    if (lseek(fdout, count, SEEK_SET) == -1 ){
        perror("Ошибка вызова функции lseek");
        exit(1);
    }
    if (write(fdout, "", 1) != 1 ){
        perror("Ошибка вызова функции write");
        exit(1);
    }
    if ((dst = mmap(0, count, PROT_READ | PROT_WRITE, MAP_SHARED, fdout, 0)) == MAP_FAILED ){
        perror("Ошибка вызова функции mmap для выходного файла");
        exit(1);
    }
    memcpy(dst, out, count);
}