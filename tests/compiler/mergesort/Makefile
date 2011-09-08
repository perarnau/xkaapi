KAAPIDIR=$(HOME)/install/xkaapi_rose
SSCC=$(KAAPIDIR)/bin/kacc
#SSCC=/usr/bin/gcc
CCFLAGS= -O3

all: main

main: main.c $(SSCC) Makefile
	$(SSCC) $(CCFLAGS) -keep main.c -o main -L$(HOME)/install/lib -lpapi

clean:
	rm -rf *.o main

run:
	./run.sh
