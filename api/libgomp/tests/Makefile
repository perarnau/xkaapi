CC=gcc-mp-4.6
CFLAGS=-g -fopenmp -fdump-tree-all -O3 -Wall
LDFLAGS=-lgomp

all: hello par-shared

hello: hello.o

par-shared: par-shared.o

hello.o: hello.c

par-shared.o: par-shared.c

clean:
	rm -rf *~ *.o hello par-shared hello.c.* par-shared.c.*

.PHONY: clean
