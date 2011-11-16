##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL, TG and VD on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

lib_LTLIBRARIES= libkaapi.la

libkaapi_la_CPPFLAGS=$(AM_CPPFLAGS) $(CPPFLAGS) -D_GNU_SOURCE $(KAAPILIB_BUILD_CPPFLAGS) -I$(srcdir)/machine/common 
libkaapi_la_CFLAGS=@AM_CFLAGS@ $(CFLAGS) $(STD_C99_FLAGS) $(KAAPILIB_BUILD_CFLAGS)
libkaapi_la_LDFLAGS = -version-info 0:0:0 $(KAAPILIB_BUILD_LDFLAGS)
libkaapi_la_LIBADD = $(KAAPILIB_BUILD_LIBS)

libkaapi_la_SOURCES=\
  misc/kaapi_version.c \
  kaapi_defs.h \
  kaapi_error.h \
  kaapi_machine.h \
  kaapi.h \
  kaapi_atomic.h \
  kaapi_impl.h \
  kaapi_format.h \
  kaapi_hashmap.h \
  kaapi_allocator.h \
  kaapi_task.h \
  kaapi_tasklist.h \
  kaapi_network.h \
  kaapi_compiler.h \
  kaapi_trace.h\
  kaapi_event.h\
  kaapi_event_recorder.h\
  kaapi_memory.h \
  kaapi_cpuset.h\
  kaapi_hws.h\
  kaapi_ws_queue.h\
  misc/kaapi_abort.c \
  misc/kaapi_allocator_allocate.c \
  misc/kaapi_cpuset2str.c \
  misc/kaapi_default_netapi.c \
  misc/kaapi_event_flush.c \
  misc/kaapi_get_self_kid.c \
  misc/kaapi_hashmap.c \
  misc/kaapi_hashmap_big.c \
  misc/kaapi_hashmap_big_find.c \
  misc/kaapi_hashmap_big_findinsert.c \
  misc/kaapi_hashmap_big_insert.c \
  misc/kaapi_hashmap_find.c \
  misc/kaapi_hashmap_findinsert.c \
  misc/kaapi_hashmap_insert.c \
  misc/kaapi_hashvalue.c \
  misc/kaapi_init.c \
  misc/kaapi_malloc_align.c \
  misc/kaapi_time_elapse.c \
  misc/kaapi_time_elapsedns.c \
  misc/kaapi_wq_init.c \
  misc/kaapi_wq_set.c \
  misc/kaapi_wq_slowpop.c \
  task/kaapi_fmt_accessors.c \
  task/kaapi_fmt_initbasic.c \
  task/kaapi_fmt_register.c \
  task/kaapi_fmt_resolvebybody.c \
  task/kaapi_fmt_resolvebyfmtid.c \
  task/kaapi_fmt_structregister.c \
  task/kaapi_fmt_taskregister.c \
  task/kaapi_fmt_taskregister_func.c \
  task/kaapi_fmt_taskregister_static.c \
  task/kaapi_task_aftersteal.c \
  task/kaapi_task_computeready.c \
  task/kaapi_task_initflag.c \
  task/kaapi_task_movealloc.c \
  task/kaapi_task_pushtaskdfg_queue.c \
  task/kaapi_task_standard.c \
  task/kaapi_task_startup_body.c \
  task/kaapi_task_steal_body.c \
  task/kaapi_task_stealready_body.c \
  thread/kaapi_tasklist.c \
  thread/kaapi_tasklist_activate.c \
  thread/kaapi_tasklist_criticalpath.c \
  thread/kaapi_thread_checkdeps.c \
  thread/kaapi_thread_clear.c \
  thread/kaapi_thread_computeready_access.c \
  thread/kaapi_thread_computereadylist.c \
  thread/kaapi_thread_dfg_print.c \
  thread/kaapi_thread_dfgdot_print.c \
  thread/kaapi_thread_execframe.c \
  thread/kaapi_thread_execframe_tasklist.c \
  thread/kaapi_thread_ispreempted.c \
  thread/kaapi_thread_pushpopframe.c \
  thread/kaapi_thread_rdl_print.c \
  thread/kaapi_thread_readylist_generic_execframe.c \
  thread/kaapi_thread_saveframe.c \
  thread/kaapi_thread_signalend_exec.c \
  thread/kaapi_version_addaccess.c \
  thread/kaapi_version_find.c \
  thread/kaapi_version_replicat.c \
  adapt/kaapi_adapt_adaptfmt.c \
  adapt/kaapi_adapt_begendadaptfmt.c \
  adapt/kaapi_adapt_finalize.c \
  adapt/kaapi_adapt_init.c \
  adapt/kaapi_adapt_mergefmt.c \
  adapt/kaapi_adapt_preempt_head.c \
  adapt/kaapi_adapt_preempt_tail.c \
  adapt/kaapi_adapt_preempt_thief.c \
  adapt/kaapi_adapt_preemptpoint.c \
  adapt/kaapi_adapt_pushstealcontext.c \
  adapt/kaapi_adapt_pushtask.c \
  adapt/kaapi_adapt_reqcommittask.c \
  adapt/kaapi_adapt_reqpushtask.c \
  adapt/kaapi_adapt_taskbegendbody.c \
  adapt/kaapi_adapt_taskbody.c \
  adapt/kaapi_adapt_taskmergebody.c \
  adapt/kaapi_adapt_taskstealbody.c \
  machine/common/kaapi_procinfo.c \
  machine/common/kaapi_procinfo.h \
  machine/common/kaapi_hwtopo.c \
  machine/common/kaapi_hw_selftopo.c \
  memory/kaapi_memory.c \
  memory/kaapi_memory_copy.c \
  memory/kaapi_memory_allocate.c \
  memory/kaapi_memory_bind.c \
  memory/kaapi_memory_find_writer.c \
  memory/kaapi_numa.c \
  memory/kaapi_mem.c \
  memory/kaapi_mem.h \
  workstealing/kaapi_sched_select_rand.c \
  workstealing/kaapi_sched_select_rand_f0.c \
  workstealing/kaapi_sched_select_workload_rand.c \
  workstealing/kaapi_sched_select_hierarchy.c\
  workstealing/kaapi_sched_stealprocessor.c \
  workstealing/kaapi_sched_stealstack.c \
  workstealing/kaapi_sched_stealframe.c \
  workstealing/kaapi_sched_stealtask.c \
  workstealing/kaapi_sched_splittask.c \
  workstealing/kaapi_sched_stealtasklist.c \
  workstealing/kaapi_sched_idle.c \
  workstealing/kaapi_sched_wakeup.c \
  workstealing/kaapi_sched_suspend.c \
  workstealing/kaapi_sched_sync.c \
  workstealing/kaapi_sched_wsqueue.c \
  hws/kaapi_hws_initialize.c \
  hws/kaapi_hws_emitsteal.c \
  hws/kaapi_hws_pushtask.c \
  hws/kaapi_hws_misc.c \
  hws/kaapi_hws_sched_sync.c \
  hws/kaapi_hws_counters.c \
  hws/kaapi_ws_queue_lifo.c \
  hws/kaapi_ws_queue_kproc.c \
  static/kaapi_stsched_tasksetstatic.c \
  static/kaapi_threadgroup_nothgrp.c\
  static/kaapi_stsched_taskrecv.c\
  static/kaapi_stsched_taskbcast.c

#   misc/kaapi_splitter.c\
   hws/kaapi_hws_splitter.c \
  hws/kaapi_hws_adaptive.c 

if KAAPI_TARGET_STATICSCHED
#Partitionning
libkaapi_la_SOURCES+=\
  memory/kaapi_memory_get_version.c \
  kaapi_partition.h\
  static/kaapi_stsched_create.c\
  static/kaapi_stsched_destroy.c\
  static/kaapi_stsched_initthread.c\
  static/kaapi_stsched_checkdeps.c\
  static/kaapi_stsched_partition.c\
  static/kaapi_stsched_tasksignal.c
endif

# static/kaapi_threadgroup_partition_barrier.c\
  static/kaapi_threadgroup_version_new.c\
  static/kaapi_threadgroup_version_find.c\
  static/kaapi_threadgroup_version_reader.c\
  static/kaapi_threadgroup_version_writer.c\
  static/kaapi_threadgroup_version_cumulwriter.c\
  static/kaapi_threadgroup_exec.c\
  static/kaapi_threadgroup_execframe.c\
  static/kaapi_threadgroup_print.c\
  static/kaapi_threadgroup_saverestore.c\
  static/kaapi_threadgroup_synchronize.c

if KAAPI_TARGET_MT
libkaapi_la_SOURCES+=\
  machine/mt/kaapi_mt_processorinit.c  \
  machine/mt/kaapi_mt_setconcurrency.c \
  machine/mt/kaapi_mt_getconcurrency.c \
  machine/mt/kaapi_mt_getcontext.c \
  machine/mt/kaapi_mt_setcontext.c \
  machine/mt/kaapi_mt_makecontext.c \
  machine/mt/kaapi_mt_sched_emitsteal.c \
  machine/mt/kaapi_mt_context_alloc.c \
  machine/mt/kaapi_mt_context_free.c \
  machine/mt/kaapi_mt_init.c \
  machine/mt/kaapi_mt_signal.c \
  machine/mt/kaapi_mt_machine.h \
  machine/mt/kaapi_mt_bitmap.h \
  machine/mt/kaapi_mt_perf.c \
  machine/mt/kaapi_mt_register_procs.c\
  machine/mt/kaapi_mt_suspendresume.c
endif

if KAAPI_TARGET_CUDA
libkaapi_la_SOURCES+=\
  machine/cuda/kaapi_cuda.h \
  machine/cuda/kaapi_cuda_error.h \
  machine/cuda/kaapi_cuda_register_procs.c \
  machine/cuda/kaapi_cuda_proc.c \
  machine/cuda/kaapi_cuda_proc.h \
  machine/cuda/kaapi_cuda_func.h \
  machine/cuda/kaapi_cuda_func.c \
  machine/cuda/kaapi_cuda_execframe.h \
  machine/cuda/kaapi_cuda_execframe.c \
  machine/cuda/kaapi_cuda_thread_execframe_tasklist.c \
  machine/cuda/kaapi_cuda_threadgroup_execframe.h \
  machine/cuda/kaapi_cuda_threadgroup_execframe.c \
  machine/cuda/kaapi_cuda_kasid.c
endif

include_HEADERS=\
  kaapi.h \
  kaapi_atomic.h\
  kaapi_error.h

if KAAPI_TARGET_CUDA
include_HEADERS+=\
  machine/cuda/kaapi_cuda_func.h
endif
