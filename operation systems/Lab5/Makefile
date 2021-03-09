cc = gcc
FLAGS = -g -Wall -Werror -Wextra

all: stt dnm

stt: libBST.so mainstt.o
		$(CC) $(FLAGS) -o stt mainstt.o -L. -lBST -Wl,-rpath,.

dnm: maindnm.o libBST.so
		$(CC) $(FLAGS) -rdynamic -o dnm maindnm.o -ldl	

mainstt.o: mainstt.c
		$(CC) -c $(FLAGS) mainstt.c

maindnm.o: maindnm.c
		$(CC) -c $(FLAGS) maindnm.c

libBST.so: BST.o
		$(CC) $(FLAGS) -shared -o libBST.so BST.o

BST.o: BST.c
		$(CC) -c -fPIC $(FLAGS) BST.c

clean:
		rm *.o stt dnm *.so