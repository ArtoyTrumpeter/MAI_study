CC = gcc

CCFLAGS = -g -std=c99 -Werror -Wextra -pedantic

OBJ = main.o sort.o queue.o

.SUFFIXES: .c .o

main: $(OBJ)
	@$(CC) $(CCFLAGS) -o main $(OBJ)

main.o: main.c sort.h queue.h

queue.o: queue.h

queue.c: queue.h

sort.c: sort.h

clean:
	@rm -r *.o main