CC = g++
CFLAGS = -g  -Wextra  -pedantic -Werror

all: gen bench

gen: test.o
	$(CC) $(CFLAGS)  -o gen test.o

bench: bench.o
	$(CC) $(CFLAGS)  -o main bench.o

bench.o: bench.cpp

test.o: test.cpp 

clean:
	@rm -r *.o main gen *.txt