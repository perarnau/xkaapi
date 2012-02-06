/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** francois.broquedis@imag.fr
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
#ifndef _KAAPI_LIBGOMP_
#define _KAAPI_LIBGOMP_

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#include "kaapi_impl.h"
#include "kaapic_impl.h"

/* barrier.c */

#define BAR_CYCLES 3
#define CACHE_LINE_SIZE 64

typedef struct gomp_barrier {
  kaapi_atomic_t __attribute__ ((aligned (CACHE_LINE_SIZE))) cycle;
  unsigned int __attribute__ ((aligned (CACHE_LINE_SIZE))) nthreads;
  char __attribute__ ((aligned (CACHE_LINE_SIZE))) count[BAR_CYCLES * CACHE_LINE_SIZE];
} gomp_barrier_t; 


struct PerTeamLocalStorage;
struct WorkShareRep;

/* Team information : common to all threads that shared a same parallel region 
   - localinfo is set to 0 at creation time
   - localinfo[tid] = &workshare of the thread tid is set in the trampoline task
   used to initialize task contexte.
   - the workshare is initialized in the workshare construct (parallel loop)
*/
typedef struct GlobalTeamInformation {
  kaapi_lock_t                 lock;       /* 1 iff work is init */
  int                          numthreads;
  kaapi_atomic_t               single_state;
  gomp_barrier_t               barrier;
  struct WorkShareRep*         localinfo[KAAPI_MAX_PROCESSOR];
} kaapi_libgomp_teaminfo_t;


/* Workshare structure
*/
typedef struct WorkShareRep {
  kaapi_atomic_t               init;       /* 1 iff work is init */
  kaapic_work_t                work;       /* last foreach loop context */
  int                          workload;   /* workload */
  struct WorkShareRep*         master;     /* master workshare */
} kaapi_libgompworkshared_t;


/** Pre-historic TLS  support for OMP */
typedef struct PerTeamLocalStorage {
  kaapi_libgompworkshared_t    workshare;  /* team workshare */
  kaapi_libgomp_teaminfo_t*    teaminfo;   /* team information */
  
  int                          threadid;
  int                          numthreads;
  int                          inside_single;
  kaapi_frame_t                frame;
} kaapi_libgompctxt_t ;


static inline kaapi_libgompctxt_t* GOMP_get_ctxtkproc( kaapi_processor_t* kproc )
{ 
  if (kproc->libgomp_tls == 0)
  {
    kaapi_libgompctxt_t* ctxt = (kaapi_libgompctxt_t*)malloc(sizeof(kaapi_libgompctxt_t));
    ctxt->threadid   = 0;
    ctxt->numthreads = 1;
    kproc->libgomp_tls = ctxt;
    return ctxt;
  }
  return (kaapi_libgompctxt_t*)kproc->libgomp_tls;
}

static inline kaapi_libgompctxt_t* GOMP_get_ctxt()
{
  return GOMP_get_ctxtkproc(kaapi_get_current_processor());
}


enum omp_task_kind
{
  GOMP_TASK_IMPLICIT,
  GOMP_TASK_IFFALSE,
  GOMP_TASK_WAITING,
  GOMP_TASK_TIED
};

/* init.c */

extern int gomp_nthreads_var;


void gomp_barrier_init (struct gomp_barrier *barrier, unsigned int num);
void gomp_barrier_destroy (struct gomp_barrier *barrier);
void gomp_barrier_wait (struct gomp_barrier *barrier);

extern void GOMP_barrier (void);

/* critical.c */

extern void GOMP_critical_start (void);
extern void GOMP_critical_end (void);
extern void GOMP_critical_name_start (void **);
extern void GOMP_critical_name_end (void **);
extern void GOMP_atomic_start (void);
extern void GOMP_atomic_end (void);

/* loop.c */

extern bool GOMP_loop_static_start (long, long, long, long, long *, long *);
extern bool GOMP_loop_dynamic_start (long, long, long, long, long *, long *);
extern bool GOMP_loop_guided_start (long, long, long, long, long *, long *);
extern bool GOMP_loop_runtime_start (long, long, long, long *, long *);

extern bool GOMP_loop_ordered_static_start (long, long, long, long,
					    long *, long *);
extern bool GOMP_loop_ordered_dynamic_start (long, long, long, long,
					     long *, long *);
extern bool GOMP_loop_ordered_guided_start (long, long, long, long,
					    long *, long *);
extern bool GOMP_loop_ordered_runtime_start (long, long, long, long *, long *);

extern bool GOMP_loop_static_next (long *, long *);
extern bool GOMP_loop_dynamic_next (long *, long *);
extern bool GOMP_loop_guided_next (long *, long *);
extern bool GOMP_loop_runtime_next (long *, long *);

extern bool GOMP_loop_ordered_static_next (long *, long *);
extern bool GOMP_loop_ordered_dynamic_next (long *, long *);
extern bool GOMP_loop_ordered_guided_next (long *, long *);
extern bool GOMP_loop_ordered_runtime_next (long *, long *);

extern void GOMP_parallel_loop_static_start (void (*)(void *), void *,
					     unsigned, long, long, long, long);
extern void GOMP_parallel_loop_dynamic_start (void (*)(void *), void *,
					     unsigned, long, long, long, long);
extern void GOMP_parallel_loop_guided_start (void (*)(void *), void *,
					     unsigned, long, long, long, long);
extern void GOMP_parallel_loop_runtime_start (void (*)(void *), void *,
					      unsigned, long, long, long);

extern void GOMP_loop_end (void);
extern void GOMP_loop_end_nowait (void);

/* loop_ull.c */

extern bool GOMP_loop_ull_static_start (bool, unsigned long long,
					unsigned long long,
					unsigned long long,
					unsigned long long,
					unsigned long long *,
					unsigned long long *);
extern bool GOMP_loop_ull_dynamic_start (bool, unsigned long long,
					 unsigned long long,
					 unsigned long long,
					 unsigned long long,
					 unsigned long long *,
					 unsigned long long *);
extern bool GOMP_loop_ull_guided_start (bool, unsigned long long,
					unsigned long long,
					unsigned long long,
					unsigned long long,
					unsigned long long *,
					unsigned long long *);
extern bool GOMP_loop_ull_runtime_start (bool, unsigned long long,
					 unsigned long long,
					 unsigned long long,
					 unsigned long long *,
					 unsigned long long *);

extern bool GOMP_loop_ull_ordered_static_start (bool, unsigned long long,
						unsigned long long,
						unsigned long long,
						unsigned long long,
						unsigned long long *,
						unsigned long long *);
extern bool GOMP_loop_ull_ordered_dynamic_start (bool, unsigned long long,
						 unsigned long long,
						 unsigned long long,
						 unsigned long long,
						 unsigned long long *,
						 unsigned long long *);
extern bool GOMP_loop_ull_ordered_guided_start (bool, unsigned long long,
						unsigned long long,
						unsigned long long,
						unsigned long long,
						unsigned long long *,
						unsigned long long *);
extern bool GOMP_loop_ull_ordered_runtime_start (bool, unsigned long long,
						 unsigned long long,
						 unsigned long long,
						 unsigned long long *,
						 unsigned long long *);

extern bool GOMP_loop_ull_static_next (unsigned long long *,
				       unsigned long long *);
extern bool GOMP_loop_ull_dynamic_next (unsigned long long *,
					unsigned long long *);
extern bool GOMP_loop_ull_guided_next (unsigned long long *,
				       unsigned long long *);
extern bool GOMP_loop_ull_runtime_next (unsigned long long *,
					unsigned long long *);

extern bool GOMP_loop_ull_ordered_static_next (unsigned long long *,
					       unsigned long long *);
extern bool GOMP_loop_ull_ordered_dynamic_next (unsigned long long *,
						unsigned long long *);
extern bool GOMP_loop_ull_ordered_guided_next (unsigned long long *,
					       unsigned long long *);
extern bool GOMP_loop_ull_ordered_runtime_next (unsigned long long *,
						unsigned long long *);

/* ordered.c */

extern void GOMP_ordered_start (void);
extern void GOMP_ordered_end (void);

/* parallel.c */

extern void GOMP_parallel_start (void (*) (void *), void *, unsigned);
extern void GOMP_parallel_end (void);

/* team.c */

extern void GOMP_task (void (*) (void *), void *, void (*) (void *, void *),
		       long, long, bool, unsigned);
extern void GOMP_taskwait (void);

/* sections.c */

extern unsigned GOMP_sections_start (unsigned);
extern unsigned GOMP_sections_next (void);
extern void GOMP_parallel_sections_start (void (*) (void *), void *,
					  unsigned, unsigned);
extern void GOMP_sections_end (void);
extern void GOMP_sections_end_nowait (void);

/* single.c */
extern bool GOMP_single_start (void);
extern void *GOMP_single_copy_start (void);
extern void GOMP_single_copy_end (void *);


#endif // #ifndef _KAAPI_LIBGOMP_
