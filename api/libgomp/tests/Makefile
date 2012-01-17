CC=gcc-mp-4.6
CFLAGS=-g -fopenmp -fdump-tree-all -O3 -Wall
LDFLAGS=-lgomp

PROGS= hello par-shared

all: $(PROGS) get-gimple

hello: hello.o

par-shared: par-shared.o

%.o: %.c
	$(CC) $(CFLAGS) -c $<

.PHONY: clean get-gimple

get-gimple: $(PROGS)
	./get-gimple.sh "$(PROGS)"

clean:
	rm -rf *~ *.o $(PROGS) hello.c.* par-shared.c.* gimp-*

