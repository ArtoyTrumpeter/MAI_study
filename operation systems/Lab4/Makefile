FLAGS=-pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -lm
COMPILLER=gcc

all: start child

start: main.o
	$(COMPILLER) $(FLAGS) -o lab4 main.o

main.o: main.c
	$(COMPILLER) -c $(FLAGS) main.c

child: child.c
	$(COMPILLER) $(FLAGS) -o child child.c

clear:
	-rm -f *.o *.gch child lab4