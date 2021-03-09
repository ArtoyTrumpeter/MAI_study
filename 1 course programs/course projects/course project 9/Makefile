CC = gcc

CFLAGS = -g -std=c99 -Wextra  -pedantic -Werror 

all: generate search sort

OBJ = generate.o my_data.o 

SCH = search.o my_data.o

MST = sort.o my_data.o

generate: $(OBJ)
	$(CC) $(CCFLAGS) -o generate $(OBJ)

search: $(SCH)
	$(CC) $(CCFLAGS) -o search $(SCH)

sort: $(MST)
	$(CC) $(CCFLAGS) -o sort $(MST)

generate.o: generate.c my_data.h

search.o: search.c my_data.h

sort.o: sort.c my_data.h

my_data.o: my_data.h

clean:
	@rm -r *.o
	@rm -r search generate sort

delete:
	@rm -r search generate sort

cleantxt:
	@rm -r *.txt