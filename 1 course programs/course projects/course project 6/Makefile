CC = gcc
CFLAGS = -g -std=c99 -Wextra  -pedantic -Werror

all: generate print execute

generate: io.o generate.o
	$(CC) $^ -o generate

print: io.o print.o
	$(CC) $^ -o print

execute: io.o execute.o
	$(CC) $^ -o execute

clean:
	rm -f execute generate print *.bin *.o