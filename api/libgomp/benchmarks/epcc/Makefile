CC = gcc
CFLAGS = -fopenmp -O3 -g -DOMPCLOCK
LDFLAGS = -lm -lgomp

TOOLS = delay.o getclock.o

all: syncbench schedbench

syncbench: $(TOOLS) syncbench.o

syncbench.o: syncbench.c

schedbench: $(TOOLS) schedbench.o

schedbench.o: schedbench.c

delay.o: delay.c

getclock.o: getclock.c



.PHONY: clean

clean:
	rm -rf syncbench schedbench *.o *~
