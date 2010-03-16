##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL, TG and VD on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

lib_LTLIBRARIES= libxkaapi.la

libxkaapi_la_CPPFLAGS=$(AM_CPPFLAGS) $(CPPFLAGS) -D_GNU_SOURCE
libxkaapi_la_CFLAGS=@AM_CFLAGS@ $(CFLAGS) $(STD_C99_FLAGS)
libxkaapi_la_LDFLAGS = -version-info 0:0:0 $(XKAAPILIB_BUILD_LDFLAGS)
libxkaapi_la_LIBADD = $(XKAAPILIB_BUILD_LIBS)

## Set up compilation of the hardware depend part
#if (KAAPI_TARGET_MT)
#  execute_process( 
#    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/machine/mt/kaapi_mt_machine.h ${CMAKE_CURRENT_BINARY_DIR}/kaapi_machine.h
#  )
#endif (KAAPI_TARGET_MT)

#deprecated   workstealing/kaapi_sched_stealtask.c 

libxkaapi_la_SOURCES=\
  misc/kaapi_version.c \
  misc/kaapi_init.c \
  misc/kaapi_time.c \
  misc/kaapi_hashvalue.c \
  misc/kaapi_hashmap.c \
  misc/kaapi_liststack.c \
  misc/kaapi_thread_clear.c \
  workstealing/kaapi_sched_select_rand_level.c \
  workstealing/kaapi_sched_select_rand_first.c \
  workstealing/kaapi_sched_select_rand.c \
  workstealing/kaapi_sched_select_workload_rand.c \
  workstealing/kaapi_sched_stealprocessor.c \
  workstealing/kaapi_sched_stealstack.c \
  workstealing/kaapi_sched_stealend.c \
  workstealing/kaapi_sched_idle.c \
  workstealing/kaapi_sched_wakeup.c \
  workstealing/kaapi_sched_suspend.c \
  workstealing/kaapi_sched_sync.c \
  task/kaapi_stack_init.c \
  task/kaapi_stack_clear.c \
  task/kaapi_stack_print.c \
  task/kaapi_stack_execframe.c \
  task/kaapi_stack_saveframe.c \
  task/kaapi_task_standard.c \
  task/kaapi_task_aftersteal.c \
  task/kaapi_task_steal.c \
  task/kaapi_task_signal.c \
  task/kaapi_task_splitter_dfg.c \
  task/kaapi_task_pushstealcontext.c\
  task/kaapi_task_finalize.c \
  task/kaapi_task_preemptpoint.c \
  task/kaapi_task_preempt_next.c \
  task/kaapi_task_preempt_reverse.c \
  task/kaapi_fmt_register.c \
  task/kaapi_fmt_taskregister.c \
  task/kaapi_fmt_structregister.c \
  task/kaapi_fmt_resolvebybody.c \
  task/kaapi_fmt_resolvebyfmtid.c \
  kaapi_datastructure.h \
  kaapi_defs.h \
  kaapi_error.h \
  kaapi_machine.h \
  kaapi.h \
  kaapi_impl.h \
  task/kaapi_stack.h \
  task/kaapi_task.h
#  task/kaapi_fmt_basictype.c

if KAAPI_TARGET_MT

libxkaapi_la_SOURCES+=\
  machine/mt/kaapi_mt_processorinit.c  \
  machine/mt/kaapi_mt_topology.c \
  machine/mt/kaapi_mt_setconcurrency.c \
  machine/mt/kaapi_mt_getconcurrency.c \
  machine/mt/kaapi_mt_getcontext.c \
  machine/mt/kaapi_mt_setcontext.c \
  machine/mt/kaapi_mt_set_workload.c \
  machine/mt/kaapi_mt_makecontext.c \
  machine/mt/kaapi_mt_selfprocessor.c \
  machine/mt/kaapi_mt_selfstack.c \
  machine/mt/kaapi_mt_sched_emitsteal.c \
  machine/mt/kaapi_mt_request_reply.c \
  machine/mt/kaapi_mt_sched_advance.c \
  machine/mt/kaapi_mt_context_alloc.c \
  machine/mt/kaapi_mt_context_free.c \
  machine/mt/kaapi_mt_init.c \
  machine/mt/kaapi_mt_machine.h \
  machine/mt/kaapi_mt_perf.c 
endif


include_HEADERS=\
	kaapi.h \
	kaapi_error.h

check_PROGRAMS=kaapi_check_header kaapi_check_header_strict
check_headers_SRC_C=\
	check/c_kaapi_header.c \
	check/c_kaapi_error_header.c
check_headers_SRC_CXX=\
	check/c++_kaapi_header.cpp \
	check/c++_kaapi_error_header.cpp
check_headers_SRC_base=\
	check/empty_main.c

kaapi_check_header_strict_CPPFLAGS=$(AM_CPPFLAGS) -Werror
kaapi_check_header_strict_CFLAGS=$(AM_CFLAGS) $(STD_C99_FLAGS)
kaapi_check_header_strict_CXXFLAGS=$(AM_CXXFLAGS) $(STD_CXX0X_FLAGS)
kaapi_check_header_strict_SOURCES=\
	$(check_headers_SRC_base)
if STD_C99
kaapi_check_header_strict_SOURCES+=\
	$(check_headers_SRC_C)
endif
if STD_CXX0X
kaapi_check_header_strict_SOURCES+=\
	$(check_headers_SRC_CXX)
endif

# No -pedantic here...
kaapi_check_header_CPPFLAGS=@AM_CPPFLAGS@ -Werror -D_KAAPI_DISABLE_WARNINGS
kaapi_check_header_SOURCES=\
	$(check_headers_SRC_C) $(check_headers_SRC_CXX) $(check_headers_SRC_base)

