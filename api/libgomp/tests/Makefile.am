
AM_CFLAGS=-fopenmp -fdump-tree-all
AM_LDFLAGS=-lgomp

PROGS= parallel shared critical nested barrier single fibo task-firstprivate loop loop2

check_PROGRAMS=$(PROGS)

TESTS_ENVIRONMENT = $(builddir)/../kaapi-gomp-wrapper --time
TESTS=$(PROGS)

TOOLBOX_SRC=test-toolbox.c test-toolbox.h

parallel_SOURCES=parallel.c $(TOOLBOX_SRC)
shared_SOURCES=shared.c $(TOOLBOX_SRC)
critical_SOURCES=critical.c $(TOOLBOX_SRC)
nested_SOURCES=nested.c $(TOOLBOX_SRC)
barrier_SOURCES=barrier.c $(TOOLBOX_SRC)
single_SOURCES=single.c $(TOOLBOX_SRC)
fibo_SOURCES=fibo.c $(TOOLBOX_SRC)
loop_SOURCES=loop.c $(TOOLBOX_SRC)
loop2_SOURCES=loop2.c $(TOOLBOX_SRC)
task_firstprivate_SOURCES=task-firstprivate.c $(TOOLBOX_SRC)

clean-local:
	$(RM) *.c.*t.*

all-local:
	find . -type f -iname "*.c.*t.*" -not -iname "*.c.*t.*omp*" -exec rm {} \;

