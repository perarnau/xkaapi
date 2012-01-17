CC=gcc-mp-4.6
CFLAGS=-g -fopenmp -fdump-tree-all -O3 -Wall
LDFLAGS=-lgomp

PROGS= hello par-shared

all: hello par-shared get-gimple

hello: hello.o

par-shared: par-shared.o

hello.o: hello.c

par-shared.o: par-shared.c

get-gimple: $(PROGS)
	./get-gimple.sh "$(PROGS)"

clean:
	rm -rf *~ *.o $(PROGS) hello.c.* par-shared.c.* gimp-*

.PHONY: clean
