CC=gcc-mp-4.6
CFLAGS=-g -fopenmp -fdump-tree-all -O3 -Wall
LDFLAGS=-lgomp

PROGS= hello par-shared critical

all: $(PROGS) get-gimple

hello: hello.o test-toolbox.o

par-shared: par-shared.o test-toolbox.o

critical: critical.o test-toolbox.o

# We don't want to generate gimple files for the test-toolbox.
test-toolbox.o: test-toolbox.c test-toolbox.h
	$(CC) -g -O3 -Wall -c $<

%.o: %.c test-toolbox.h
	$(CC) $(CFLAGS) -c $<

.PHONY: clean get-gimple

get-gimple: $(PROGS)
	./get-gimple.sh "$(PROGS)"

clean:
	rm -rf *~ *.o $(PROGS) hello.c.* par-shared.c.* gimp-*

