##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL, TG and VD on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# Add the binary directory for generated headers :
include_directories (${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

#################
## LIBRARY BEG ##

# Set up compilation of the hardware depend part
if (KAAPI_TARGET_MT)
  execute_process(
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/machine/mt/kaapi_mt_machine.h ${CMAKE_CURRENT_BINARY_DIR}/kaapi_machine.h
  )
#elif (KAAPI_TARGET_GPU)
#  execute_process( 
#      ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/machine/gpu/kaapi_gpu_machine.h ${CMAKE_CURRENT_BINARY_DIR}/kaapi_machine.h
#  )
#elif (KAAPI_MPSOC_ST)
#  execute_process( 
#      ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/machine/mpsoc/kaapi_gpu_machine.h ${CMAKE_CURRENT_BINARY_DIR}/kaapi_machine.h
#  )
endif (KAAPI_TARGET_MT)

# Shared lib :
add_library (xkaapi SHARED 
  workstealing/kaapi_sched_select_rand_level.c
  workstealing/kaapi_sched_select_rand_first.c
  workstealing/kaapi_sched_select_rand.c
  workstealing/kaapi_sched_stealprocessor.c
  workstealing/kaapi_sched_stealstack.c
  workstealing/kaapi_sched_stealtask.c
  workstealing/kaapi_sched_idle.c
  workstealing/kaapi_sched_wakeup.c
  workstealing/kaapi_sched_suspend.c
  workstealing/kaapi_sched_sync.c
  task/kaapi_stack_init.c
  task/kaapi_stack_clear.c
  task/kaapi_stack_print.c
  task/kaapi_stack_execall.c
  task/kaapi_stack_execchild.c
  task/kaapi_task_standard.c
  task/kaapi_task_aftersteal.c
  task/kaapi_task_steal.c
  task/kaapi_task_splitter_dfg.c
  task/kaapi_task_finalize.c
  task/kaapi_task_preempt.c
  task/kaapi_fmt_register.c
  task/kaapi_fmt_taskregister.c
  task/kaapi_fmt_structregister.c
  task/kaapi_fmt_resolvebybody.c
  task/kaapi_fmt_resolvebyfmtid.c
#  task/kaapi_fmt_basictype.c
  machine/mt/kaapi_mt_processorinit.c
  machine/mt/kaapi_mt_topology.c
  machine/mt/kaapi_mt_setconcurrency.c
  machine/mt/kaapi_mt_getconcurrency.c
  machine/mt/kaapi_mt_getcontext.c
  machine/mt/kaapi_mt_setcontext.c
  machine/mt/kaapi_mt_makecontext.c
  machine/mt/kaapi_mt_selfprocessor.c
  machine/mt/kaapi_mt_selfstack.c
  machine/mt/kaapi_mt_sched_emitsteal.c
  machine/mt/kaapi_mt_request_reply.c
  machine/mt/kaapi_mt_task_signal.c
  machine/mt/kaapi_mt_sched_advance.c
  machine/mt/kaapi_mt_context_alloc.c
  machine/mt/kaapi_mt_context_free.c
  machine/mt/kaapi_mt_init.c
  misc/kaapi_init.c
  misc/kaapi_time.c
  misc/kaapi_hashvalue.c
)

# Version number :
set_target_properties(xkaapi PROPERTIES VERSION 0.0.1 SOVERSION 0)

# Static lib :
add_library (xkaapi-static STATIC 
  workstealing/kaapi_sched_select_rand_level.c
  workstealing/kaapi_sched_select_rand_first.c
  workstealing/kaapi_sched_select_rand.c
  workstealing/kaapi_sched_stealprocessor.c
  workstealing/kaapi_sched_stealstack.c
  workstealing/kaapi_sched_stealtask.c
  workstealing/kaapi_sched_idle.c
  workstealing/kaapi_sched_wakeup.c
  workstealing/kaapi_sched_suspend.c
  workstealing/kaapi_sched_sync.c
  task/kaapi_stack_init.c
  task/kaapi_stack_clear.c
  task/kaapi_stack_print.c
  task/kaapi_stack_execall.c
  task/kaapi_stack_execchild.c
  task/kaapi_task_standard.c
  task/kaapi_task_aftersteal.c
  task/kaapi_task_steal.c
  task/kaapi_task_splitter_dfg.c
  task/kaapi_task_finalize.c
  task/kaapi_task_preempt.c
  task/kaapi_fmt_register.c
  task/kaapi_fmt_taskregister.c
  task/kaapi_fmt_structregister.c
  task/kaapi_fmt_resolvebybody.c
  task/kaapi_fmt_resolvebyfmtid.c
#  task/kaapi_fmt_basictype.c
  machine/mt/kaapi_mt_processorinit.c
  machine/mt/kaapi_mt_topology.c
  machine/mt/kaapi_mt_setconcurrency.c
  machine/mt/kaapi_mt_getconcurrency.c
  machine/mt/kaapi_mt_getcontext.c
  machine/mt/kaapi_mt_setcontext.c
  machine/mt/kaapi_mt_makecontext.c
  machine/mt/kaapi_mt_selfprocessor.c
  machine/mt/kaapi_mt_selfstack.c
  machine/mt/kaapi_mt_sched_emitsteal.c
  machine/mt/kaapi_mt_request_reply.c
  machine/mt/kaapi_mt_task_signal.c
  machine/mt/kaapi_mt_sched_advance.c
  machine/mt/kaapi_mt_context_alloc.c
  machine/mt/kaapi_mt_context_free.c
  machine/mt/kaapi_mt_init.c
  misc/kaapi_init.c
  misc/kaapi_time.c
  misc/kaapi_hashvalue.c
)

set_target_properties(xkaapi-static PROPERTIES OUTPUT_NAME "xkaapi")

# These next two lines are required but it is unclear exactly what they do.
# The CMake FAQ mentions they are necessary and it does not work otherwise.
set_target_properties(xkaapi PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(xkaapi-static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

## LIBRARY END ##
#################


########################
## CHECK INCLUDES BEG ##

# Add "CheckIncludeFiles" CMake module :
include (CheckIncludeFiles)

# Check for some headers :
check_include_files (errno.h      HAVE_ERRNO_H     )
check_include_files (limits.h     HAVE_LIMITS_H    )
check_include_files (numa.h       HAVE_NUMA_H      )
check_include_files (pthread.h    HAVE_PTHREAD_H   )
check_include_files (sched.h      HAVE_SCHED_H     )
check_include_files (stdlib.h     HAVE_STDLIB_H    )
check_include_files (sys/time.h   HAVE_SYS_TIME_H  )
check_include_files (sys/sysctl.h HAVE_SYS_SYSCTL_H)
check_include_files (unistd.h     HAVE_UNISTD_H    )

# Context switch :
check_include_files (setjmp.h     HAVE_SETJMP_H    )
check_include_files (ucontext.h   HAVE_UCONTEXT_H  )

# For debug mode :
check_include_files (stdio.h      HAVE_STDIO_H     )

## CHECK INCLUDES END ##
########################


#######################
## SET CONSTANTS BEG ##

#Maximum number of architecture depending on the installation (1 or 2 in case of mixing CPU/GPU, 3 in case of mixing CPU/GPU/MPSoC)
# This number may be also depend on the caracteristic of the processors.... and one could consider having N different architectures
# For each architecture, task should have a specific entry point
# This will lead to incompatible binary library and should be visible in the
# library name
set(KAAPI_MAX_ARCH 3)      

# Get the number of online processors using getconf :
execute_process (COMMAND getconf _NPROCESSORS_ONLN OUTPUT_VARIABLE KAAPI_MAX_PROCESSOR)

# Get minimum stack size from pthread constant :
try_run (
  KAAPI_STACK_MIN_RUN_RESULT
  KAAPI_STACK_MIN_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/config/check_pthread_stack_min.c
  RUN_OUTPUT_VARIABLE KAAPI_STACK_MIN )

# TODO: This will break with cross-compilation
# Get the cache line size :
if (CMAKE_HOST_APPLE)
  execute_process (
    COMMAND sysctl -n hw.cachelinesize
    RESULT_VARIABLE KAAPI_CACHE_LINE_RUN_RESULT
    OUTPUT_VARIABLE KAAPI_CACHE_LINE
  )
else (CMAKE_HOST_APPLE)
endif (CMAKE_HOST_APPLE)

if (KAAPI_CACHE_LINE)
  message ("++ Cache linesize found")
else (KAAPI_CACHE_LINE)
  message ("++ Cache linesize not found, use 64")
  set(KAAPI_CACHE_LINE 64)
endif (KAAPI_CACHE_LINE)


# Get maximal data constraint alignment for natural type, assume it is double or void* on 64bits machine
set( KAAPI_MAX_DATA_ALIGNMENT 8)

## SET CONSTANTS END ##
#######################


########################
## CHECK FUNCTION BEG ##

include (CheckFunctionExists)

set (CMAKE_REQUIRED_INCLUDES "/usr/include")
set (CMAKE_REQUIRED_LIBRARIES "pthread")
check_function_exists ("pthread_attr_setaffinity_np" KAAPI_USE_SCHED_AFFINITY)

if (KAAPI_USE_SCHED_AFFINITY)
  message ("++ X-Kaapi configured for affinity")
endif (KAAPI_USE_SCHED_AFFINITY)

## CHECK FUNCTION END ##
########################

include (CheckTypeSize)

###################
## void* BEG ##

check_type_size ("void*" KAAPI_SIZEOF_VOIDP)

## CPU_SET_T END ##
###################


###################
## CPU_SET_T BEG ##

# Cpu_set_t :
set (CMAKE_EXTRA_INCLUDE_FILES "sched.h")
check_type_size ("cpu_set_t" CPU_SET_T)

## CPU_SET_T END ##
###################


##################################
## CONFIGURE KAAPI_CONFIG.H BEG ##

# Generate kaapi_config.h from kaapi_config.h.in :
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/kaapi_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/kaapi_config.h)

## CONFIGURE KAAPI_CONFIG.H END ##
##################################


###################################
## SET SIZEOF PRIVATE STRUCT BEG ##

set (CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_BINARY_DIR})
set (CMAKE_EXTRA_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kaapi_private_structure.h)

set (CMAKE_EXTRA_INCLUDE_FILES)

## set SIZEOF PRIVATE STRUCT END ##
###################################


#######################
## INSTALL RULES BEG ##

find_library(PTHREAD_LIBRARY pthread)

target_link_libraries(xkaapi ${PTHREAD_LIBRARY})
target_link_libraries(xkaapi-static ${PTHREAD_LIBRARY})

if (HAVE_NUMA_H)
  find_library(NUMA_LIBRARY numa)
  target_link_libraries(xkaapi ${NUMA_LIBRARY})
endif (HAVE_NUMA_H)

install(TARGETS xkaapi xkaapi-static
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/kaapi.h ${CMAKE_CURRENT_SOURCE_DIR}/kaapi_error.h DESTINATION include/)

## INSTALL RULES BEG ##
#######################
