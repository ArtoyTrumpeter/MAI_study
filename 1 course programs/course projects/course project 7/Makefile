CC = gcc

CFLAGS = -g -std=c99 -Wextra  -pedantic -Werror 

OBJ = main.o simetric_vector.o function.o

main: $(OBJ)
	$(CC) $(CCFLAGS) -o main $(OBJ)

main.o: main.c complex.h 

function.c:complex.h

simetric_vector.c:complex.h

clean:
	@rm -r *.o main