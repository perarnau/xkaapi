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
  workstealing/kaapi_sched_idle.c \
  workstealing/kaapi_sched_wakeup.c \
  workstealing/kaapi_sched_suspend.c \
  workstealing/kaapi_sched_sync.c \
  workstealing/kaapi_sched_criticalsteal.c \
  task/kaapi_stack_init.c \
  task/kaapi_stack_clear.c \
  task/kaapi_stack_print.c \
  task/kaapi_stack_execframe.c \
  task/kaapi_stack_saveframe.c \
  task/kaapi_task_standard.c \
  task/kaapi_task_aftersteal.c \
  task/kaapi_task_steal.c \
  task/kaapi_task_splitter_dfg.c \
  task/kaapi_task_splitter_adapt.c \
  task/kaapi_task_pushstealcontext.c\
  task/kaapi_task_adapt_allocateresult.c\
  task/kaapi_task_adapt_deallocateresult.c\
  task/kaapi_task_reqreply.c \
  task/kaapi_task_preempt_head.c \
  task/kaapi_task_preempt_tail.c \
  task/kaapi_task_preempt_thief.c \
  task/kaapi_task_preemptpoint.c \
  task/kaapi_task_finalize.c \
  task/kaapi_task_return.c \
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
  static/kaapi_threadgroup_checkdeps.c\
  static/kaapi_threadgroup_version.c\
  static/kaapi_threadgroup_veralloc.c\
  static/kaapi_threadgroup_taskbcast.c\
  static/kaapi_threadgroup_taskrecv.c\
  static/kaapi_threadgroup_taskrecvbcast.c\
  static/kaapi_threadgroup_tasksignal.c\
  static/kaapi_threadgroup_create.c\
  static/kaapi_threadgroup_destroy.c\
  static/kaapi_threadgroup_partition.c\
  static/kaapi_threadgroup_exec.c\
  static/kaapi_threadgroup_saverestore.c\
  static/kaapi_threadgroup_print.c\
  static/kaapi_threadgroup_vect.c\
  kaapi_staticsched.h \
  machine/common/kaapi_procinfo.c


#  task/kaapi_fmt_basictype.c

if KAAPI_TARGET_MT

libxkaapi_la_SOURCES+=\
  machine/mt/kaapi_mt_processorinit.c  \
  machine/mt/kaapi_mt_topology.c \
  machine/mt/kaapi_mt_setconcurrency.c \
  machine/mt/kaapi_mt_getconcurrency.c \
  machine/mt/kaapi_mt_getcontext.c \
  machine/mt/kaapi_mt_setcontext.c \
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
  machine/mt/kaapi_mt_perf.c \
  machine/mt/kaapi_mt_register_procs.c \
  machine/mt/kaapi_mt_processor.c
endif

if KAAPI_USE_CUDA
libxkaapi_la_SOURCES+=\
  machine/cuda/kaapi_cuda.h \
  machine/cuda/kaapi_cuda_error.h \
  machine/cuda/kaapi_cuda_register_procs.h \
  machine/cuda/kaapi_cuda_register_procs.c \
  machine/cuda/kaapi_cuda_proc.c \
  machine/cuda/kaapi_cuda_proc.h \
  machine/cuda/kaapi_cuda_func.h \
  machine/cuda/kaapi_cuda_func.c
endif

include_HEADERS=\
  kaapi.h \
  kaapi_error.h

if KAAPI_USE_CUDA
include_HEADERS+=\
  machine/cuda/kaapi_cuda_func.h
endif
