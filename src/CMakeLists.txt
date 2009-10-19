##
##  CMakeLists.txt
##  ckaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# Add the binary directory for generated headers :
include_directories (${CMAKE_CURRENT_BINARY_DIR})

# Global definitions for compile comand :
add_definitions ( "-Wall -std=c99 -pedantic -g" )

#################
## LIBRARY BEG ##

# Shared lib :
add_library (ckaapi SHARED 
  kaapi_atomic.h kaapi_attr_affinity.c kaapi_attr_destroy.c
  kaapi_attr_detach.c kaapi_attr_inheritedsched.c kaapi_attr_init.c
  kaapi_attr_schedparam.c kaapi_attr_schedpolicy.c kaapi_attr_scope.c
  kaapi_attr_stack.c kaapi_attr_stackaddr.c kaapi_attr_stacksize.c
  kaapi_cond_attr.c kaapi_cond_broadcast.c kaapi_cond_destroy.c
  kaapi_cond_init.c kaapi_cond_signal.c kaapi_cond_timedwait.c
  kaapi_cond_wait.c kaapi_init.c kaapi_mutex_attr.c
  kaapi_mutex_destroy.c kaapi_mutex_init.c kaapi_mutex_lock.c
  kaapi_mutex_trylock.c kaapi_mutex_unlock.c kaapi_sched_alloc.c
  kaapi_sched_concurrency.c kaapi_sched_run.c
  kaapi_sched_steal_thread.c kaapi_sched_steal_inside_thread.c kaapi_sched_suspend.c kaapi_sched_idle.c
  kaapi_sched_term.c kaapi_specific_create.c kaapi_specific_delete.c
  kaapi_specific_get.c kaapi_specific_set.c kaapi_thread_create.c
  kaapi_thread_detach.c kaapi_thread_equal.c kaapi_thread_exit.c
  kaapi_thread_join.c kaapi_thread_once.c kaapi_thread_self.c
  kaapi_thread_yield.c kaapi_stealapi.h
  kaapi_stealapi_processor_init.c kaapi_stealapi_request_wait.c
  kaapi_stealapi_stealcontext_init.c kaapi_stealapi_stealcontext_finalize.c
  kaapi_stealapi_processor_selectvictim.c
  kaapi_stealapi_stealcontext_pushpop.c kaapi_stealapi_init.c
  kaapi_stealapi_processor_steal.c kaapi_stealapi_synchro.h
  kaapi_stealapi_processor_alloc.c kaapi_stealapi_request_post.c
)

# Version number :
set_target_properties(ckaapi PROPERTIES VERSION 1.0.1 SOVERSION 1)

# Static lib :
add_library (ckaapi-static STATIC 
  kaapi_atomic.h kaapi_attr_affinity.c kaapi_attr_destroy.c
  kaapi_attr_detach.c kaapi_attr_inheritedsched.c kaapi_attr_init.c
  kaapi_attr_schedparam.c kaapi_attr_schedpolicy.c kaapi_attr_scope.c
  kaapi_attr_stack.c kaapi_attr_stackaddr.c kaapi_attr_stacksize.c
  kaapi_cond_attr.c kaapi_cond_broadcast.c kaapi_cond_destroy.c
  kaapi_cond_init.c kaapi_cond_signal.c kaapi_cond_timedwait.c
  kaapi_cond_wait.c kaapi_init.c kaapi_mutex_attr.c
  kaapi_mutex_destroy.c kaapi_mutex_init.c kaapi_mutex_lock.c
  kaapi_mutex_trylock.c kaapi_mutex_unlock.c kaapi_sched_alloc.c
  kaapi_sched_concurrency.c kaapi_sched_run.c
  kaapi_sched_steal_thread.c kaapi_sched_steal_inside_thread.c kaapi_sched_suspend.c kaapi_sched_idle.c
  kaapi_sched_term.c kaapi_specific_create.c kaapi_specific_delete.c
  kaapi_specific_get.c kaapi_specific_set.c kaapi_thread_create.c
  kaapi_thread_detach.c kaapi_thread_equal.c kaapi_thread_exit.c
  kaapi_thread_join.c kaapi_thread_once.c kaapi_thread_self.c
  kaapi_thread_yield.c kaapi_stealapi.h
  kaapi_stealapi_processor_init.c kaapi_stealapi_request_wait.c
  kaapi_stealapi_stealcontext_init.c kaapi_stealapi_stealcontext_finalize.c
  kaapi_stealapi_processor_selectvictim.c
  kaapi_stealapi_stealcontext_pushpop.c kaapi_stealapi_init.c
  kaapi_stealapi_processor_steal.c kaapi_stealapi_synchro.h
  kaapi_stealapi_processor_alloc.c kaapi_stealapi_request_post.c
)

set_target_properties(ckaapi-static PROPERTIES OUTPUT_NAME "ckaapi")

# These next two lines are required but it is unclear exactly what they do.
# The CMake FAQ mentions they are necessary and it does not work otherwise.
set_target_properties(ckaapi PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(ckaapi-static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

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

# Get the number of online processors using getconf :
execute_process (COMMAND getconf _NPROCESSORS_ONLN OUTPUT_VARIABLE KAAPI_MAX_PROCESSOR)

# Get minimum stack size from pthread constant :
try_run (
  KAAPI_STACK_MIN_RUN_RESULT
  KAAPI_STACK_MIN_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/config/check_pthread_stack_min.c
  RUN_OUTPUT_VARIABLE KAAPI_STACK_MIN )

# Get maximum number of keys from pthread constant :
try_run (
  KAAPI_KEYS_MAX_RUN_RESULT
  KAAPI_KEYS_MAX_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/config/check_pthread_max_keys.c
  RUN_OUTPUT_VARIABLE KAAPI_KEYS_MAX )

# Get number of iteration for dataspecific desctructors from pthread constant :
try_run (
  KAAPI_DESTRUCTOR_ITERATIONS_RUN_RESULT
  KAAPI_DESTRUCTOR_ITERATIONS_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/config/check_pthread_destructor_iterations.c
  RUN_OUTPUT_VARIABLE KAAPI_DESTRUCTOR_ITERATIONS )

## SET CONSTANTS END ##
#######################


########################
## CHECK FUNCTION BEG ##

include (CheckFunctionExists)

set (CMAKE_REQUIRED_INCLUDES "/usr/include")
set (CMAKE_REQUIRED_LIBRARIES "pthread")
check_function_exists ("pthread_attr_setaffinity_np" KAAPI_USE_SCHED_AFFINITY)

if (KAAPI_USE_SCHED_AFFINITY)
  message ("++ CKaapi configured for affinity")
endif (KAAPI_USE_SCHED_AFFINITY)

## CHECK FUNCTION END ##
########################


###################
## CPU_SET_T BEG ##

include (CheckTypeSize)

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


#########################
## SET INITIALIZER BEG ##

# Mutex Initializer :
try_run (
  KAAPI_MUTEX_INITIALIZER_RUN_RESULT
  KAAPI_MUTEX_INITIALIZER_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/config/mutex_initializer.c
  COMPILE_DEFINITIONS "-I${CMAKE_BINARY_DIR}/src -I${CMAKE_SOURCE_DIR}/src"
  RUN_OUTPUT_VARIABLE KAAPI_MUTEX_INITIALIZER)

if (KAAPI_MUTEX_INITIALIZER_RUN_RESULT)
  message ("++ Creating Mutex Fast Initializer - done")
else (KAAPI_MUTEX_INITIALIZER_RUN_RESULT)
  message ("++ Creating Mutex Fast Initializer - failed")
endif (KAAPI_MUTEX_INITIALIZER_RUN_RESULT)

# Condition Initializer :
try_run (
  KAAPI_COND_INITIALIZER_RUN_RESULT
  KAAPI_COND_INITIALIZER_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/config/cond_initializer.c
  COMPILE_DEFINITIONS "-I${CMAKE_BINARY_DIR}/src -I${CMAKE_SOURCE_DIR}/src"
  RUN_OUTPUT_VARIABLE KAAPI_COND_INITIALIZER)

if (KAAPI_COND_INITIALIZER_RUN_RESULT)
  message ("++ Creating Condition Fast Initializer - done")
else (KAAPI_COND_INITIALIZER_RUN_RESULT)
  message ("++ Creating Condition Fast Initializer - failed")
endif (KAAPI_COND_INITIALIZER_RUN_RESULT)

# Once Initializer :
try_run (
  KAAPI_ONCE_INITIALIZER_RUN_RESULT
  KAAPI_ONCE_INITIALIZER_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/config/once_initializer.c
  COMPILE_DEFINITIONS "-I${CMAKE_BINARY_DIR}/src -I${CMAKE_SOURCE_DIR}/src"
  RUN_OUTPUT_VARIABLE KAAPI_ONCE_INIT)

if (KAAPI_ONCE_INITIALIZER_RUN_RESULT)
  message ("++ Creating Once Fast Initializer - done")
else (KAAPI_ONCE_INITIALIZER_RUN_RESULT)
  message ("++ Creating Once Fast Initializer - failed")
endif (KAAPI_ONCE_INITIALIZER_RUN_RESULT)


## SET INITIALIZER END ##
#########################


###########################
## CONFIGURE KAAPI.H BEG ##

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/kaapi.h.in ${CMAKE_CURRENT_BINARY_DIR}/kaapi.h)

## CONFIGURE KAAPI.H END ##
###########################


###################################
## SET SIZEOF PRIVATE STRUCT BEG ##

set (CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_BINARY_DIR})
set (CMAKE_EXTRA_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kaapi_private_structure.h)

# Mutex :
check_type_size ("kaapi_mutex_t" KAAPI_SIZEOF_MUTEX_T)
check_type_size ("kaapi_mutexattr_t" KAAPI_SIZEOF_MUTEXATTR_T)

# Condition :
check_type_size ("kaapi_cond_t" KAAPI_SIZEOF_COND_T)
check_type_size ("kaapi_condattr_t" KAAPI_SIZEOF_CONDATTR_T)

# Attribut :
check_type_size ("kaapi_attr_t" KAAPI_SIZEOF_ATTR_T)

# Attribut :
check_type_size ("kaapi_once_t" KAAPI_SIZEOF_ONCE_T)

set (CMAKE_EXTRA_INCLUDE_FILES)

## set SIZEOF PRIVATE STRUCT END ##
###################################


################################
## CONFIGURE KAAPI_TYPE.H BEG ##

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/kaapi_type.h.in ${CMAKE_CURRENT_BINARY_DIR}/kaapi_type.h)

## CONFIGURE KAAPI_TYPE.H END ##
################################


#######################
## INSTALL RULES BEG ##

find_library(PTHREAD_LIBRARY pthread)

target_link_libraries(ckaapi ${PTHREAD_LIBRARY})
target_link_libraries(ckaapi-static ${PTHREAD_LIBRARY})

if (HAVE_NUMA_H)
  find_library(NUMA_LIBRARY numa)
  target_link_libraries(ckaapi ${NUMA_LIBRARY})
endif (HAVE_NUMA_H)

install(TARGETS ckaapi ckaapi-static
  LIBRARY DESTINATION local/lib
  ARCHIVE DESTINATION local/lib/static)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/kaapi.h ${CMAKE_CURRENT_BINARY_DIR}/kaapi_type.h DESTINATION local/include/)

## INSTALL RULES BEG ##
#######################
