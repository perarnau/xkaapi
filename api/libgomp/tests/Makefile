CC=gcc-mp-4.6
CFLAGS=-g -fopenmp -fdump-tree-all -O3 -Wall
LDFLAGS=-lgomp

PROGS= parallel shared critical nested barrier single

all: $(PROGS) get-gimple

parallel: parallel.o test-toolbox.o

shared: shared.o test-toolbox.o

critical: critical.o test-toolbox.o

nested: nested.o test-toolbox.o

barrier: barrier.o test-toolbox.o

single: single.o test-toolbox.o

# We don't want to generate gimple files for the test-toolbox.
test-toolbox.o: test-toolbox.c test-toolbox.h
	$(CC) -g -O3 -Wall -c $<

%.o: %.c test-toolbox.h
	$(CC) $(CFLAGS) -c $<

.PHONY: clean get-gimple

get-gimple: $(PROGS)
	./get-gimple.sh "$(PROGS)"

clean:
	rm -rf *~ *.o $(PROGS) parallel.c.* shared.c.* critical.c.* nested.c.* barrier.c.* single.c.* gimp-*

