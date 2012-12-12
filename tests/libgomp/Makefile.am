
DIST_SUBDIRS=gcc-testsuite

clean-recursive: SUBDIRS+=$(DIST_SUBDIRS)

AM_CFLAGS=@AM_CFLAGS@ -fopenmp
LIBS=@LIBS@ -lgomp

PROGS= parallel \
       shared   \
       critical \
       nested   \
       barrier  \
       barrier2 \
       single   \
       fibo     \
       task-firstprivate  \
       loop     \
       loop2    \
       ordered get_level  \
       get_active_level   \
       get_ancestor_thread_num
#loop3 loop4

noinst_PROGRAMS=$(PROGS)

TESTS_ENVIRONMENT = $(top_builddir)/api/libgomp/kaapi-gomp-wrapper
TESTS=$(PROGS)

TOOLBOX_SRC=test-toolbox.c test-toolbox.h

parallel_SOURCES=parallel.c $(TOOLBOX_SRC)
shared_SOURCES=shared.c $(TOOLBOX_SRC)
critical_SOURCES=critical.c $(TOOLBOX_SRC)
nested_SOURCES=nested.c $(TOOLBOX_SRC)
barrier_SOURCES=barrier.c $(TOOLBOX_SRC)
barrier2_SOURCES=barrier2.c $(TOOLBOX_SRC)
single_SOURCES=single.c $(TOOLBOX_SRC)
fibo_SOURCES=fibo.c $(TOOLBOX_SRC)
loop_SOURCES=loop.c $(TOOLBOX_SRC)
loop2_SOURCES=loop2.c $(TOOLBOX_SRC)
loop3_SOURCES=loop3.c $(TOOLBOX_SRC)
ordered_SOURCES=ordered.c $(TOOLBOX_SRC)
get_level_SOURCES=get_level.c $(TOOLBOX_SRC)
get_active_level_SOURCES=get_active_level.c $(TOOLBOX_SRC)
get_ancestor_thread_num_SOURCES=get_ancestor_thread_num.c $(TOOLBOX_SRC)

loop4_SOURCES=loop4.c $(TOOLBOX_SRC)
loop4_CPPFLAGS=-I$(top_srcdir)/api/libgomp
loop4_LDFLAGS=-L$(top_builddir)/api/libgomp -lomp_ext

task_firstprivate_SOURCES=task-firstprivate.c $(TOOLBOX_SRC)

