/*
** kaapi_config.h.in
** ckaapi
** 
** Created on Tue Mar 31 15:19:34 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#ifndef _KAAPI_CONFIG_H
#define _KAAPI_CONFIG_H 1


/* ================================= Headers ================================ */

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 if you have the <numa.h> header file. */
/* #undef HAVE_NUMA_H */

/* Define to 1 if you have the <pthread.h> header file. */
#define HAVE_PTHREAD_H 1

/* Define to 1 if you have the <sched.h> header file. */
#define HAVE_SCHED_H 1

/* Define to 1 if you have the <setjmp.h> header file. */
#define HAVE_SETJMP_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/sysctl.h> header file. */
#define HAVE_SYS_SYSCTL_H 1

/* Define to 1 if you have the <ucontext.h> header file. */
#define HAVE_UCONTEXT_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* ========================================================================== */

/* Define to 1 if you have the "cpu_set_t" type. */
/* #undef HAVE_CPU_SET_T */

#ifdef HAVE_CPU_SET_T
#  define _GNU_SOURCE
#endif

/* ================================== Archi ================================= */

/* Define the number of online processors. */
#define KAAPI_MAX_PROCESSOR 2


/* Define the maximum number of keys. */
#define KAAPI_KEYS_MAX 512


/* Define if Kaapi uses sched affinity. */
/* #undef KAAPI_USE_SCHED_AFFINITY */

#ifdef __linux__
#  define KAAPI_USE_LINUX 1
#  ifdef HAVE_UCONTEXT_H
#    define KAAPI_USE_UCONTEXT
#  elif HAVE_SETJMP_H
#    error "Not implemented yet"
#    define KAAPI_USE_SETJMP
#  endif
#endif

#ifdef __APPLE__
#  define KAAPI_USE_APPLE 1
#  ifdef HAVE_SETJMP_H
#    define KAAPI_USE_SETJMP
#  elif HAVE_UCONTEXT_H
#    define KAAPI_USE_UCONTEXT
#  endif
#endif

#ifdef __i386__
#  define KAAPI_USE_ARCH_X86 1
#endif

#ifdef __x86_64
#  define KAAPI_USE_ARCH_X86_64 1
#  define KAAPI_USE_ARCH_X86 1
#endif

#ifdef __ia64__
#  define KAAPI_USE_ARCH_IA64 1
#endif

#ifdef __PPC
#  define KAAPI_USE_ARCH_PPC 1
#endif

#ifdef __arm
#  define KAAPI_USE_ARCH_ARM 1
#endif
 
#ifdef __sparc__
#  error "Unsupported Architecture"
#endif

#ifdef __mips
#  error "Unsupported Architecture"
#endif

/* ========================================================================== */

#endif // _KAAPI_CONFIG_H
