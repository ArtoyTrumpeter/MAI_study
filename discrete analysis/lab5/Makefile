NODEBUG = -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result -pedantic
DEBUG = -std=c++11 -O0 -g -Wextra -Wall -Wno-sign-compare -Wno-unused-result -pedantic

CXX = g++
CXXFLAGS = $(NODEBUG)
NAME = solution

all: main

main: tree main.cpp
	$(CXX) $(CXXFLAGS) main.cpp *.o -o $(NAME)

tree: SuffTree.h SuffTree.cpp
	$(CXX) $(CXXFLAGS) SuffTree.cpp -c

clean:
	rm -f *.o $(NAME)