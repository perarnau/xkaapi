/*
 ** xkaapi
 ** 
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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
#ifndef KAAPIC_HIMPL_INCLUDED
# define KAAPIC_HIMPL_INCLUDED

/* kaapic_save,restore_frame */
#include "kaapi_impl.h"
#include "kaapic.h"

/* implementation for kaapic API */
#if defined(__cplusplus)
extern "C" {
#endif

#define KAAPIC_USE_KPROC_LOCK 1

extern void _kaapic_register_task_format(void);

/* closure for the body of the for each */
typedef struct kaapic_body_arg_t {
  union {
    void (*f_c)(int32_t, int32_t, int32_t, ...);
    void (*f_f)(int32_t*, int32_t*, int32_t*, ...);
  } u;
  unsigned int        nargs;
  void*               args[];
} kaapic_body_arg_t;


/* Signature of foreach body 
   Called with (first, last, tid, arg) in order to do
   computation over the range [first,last[.
*/
typedef void (*kaapic_foreach_body_t)(int32_t, int32_t, int32_t, kaapic_body_arg_t* );


/* Default attribut if not specified
*/
extern kaapic_foreach_attr_t kaapic_default_attr;


/* exported foreach interface 
   evaluate body_f(first, last, body_args) in parallel, assuming
   that the evaluation of body_f(i, j, body_args) does not impose
   dependency with evaluation of body_f(k,l, body_args) if [i,j[ and [k,l[
   does not intersect.
*/
extern int kaapic_foreach_common
(
  kaapi_workqueue_index_t first,
  kaapi_workqueue_index_t last,
  kaapic_foreach_attr_t*  attr,
  kaapic_foreach_body_t   body_f,
  kaapic_body_arg_t*      body_args
);


/* wrapper for kaapic_foreach(...) and kaapic_foreach_withformat(...)
*/
extern void kaapic_foreach_body2user(
  int32_t first, 
  int32_t last, 
  int32_t tid, 
  kaapic_body_arg_t* arg 
);


/*
*/
typedef struct kaapic_arg_info_t
{
  kaapi_access_mode_t mode;
  kaapi_memory_view_t view;
  const struct kaapi_format_t* format;

  /* kaapi versionning for shared pointer 
     also used to store address of union 'value' 
     for by-value argument.
     Currenlty value are copied into the version field
     of the access.
  */
  kaapi_access_t access;

} kaapic_arg_info_t;


/*
*/
typedef struct kaapic_task_info
{
  void             (*body)();
  uintptr_t         nargs;
  kaapic_arg_info_t args[1];
} kaapic_task_info_t;


/*
*/
extern int kaapic_spawn_ti(
  kaapi_thread_t* thread, 
  kaapi_task_body_t body, 
  kaapic_task_info_t* ti
);


/* work array distribution. allow for random access. 
   The thread tid has a reserved slice [first, last)
   iff map[tid] != 0.
   Then it should process work on its slice, where:
     first = startindex[tid2pos[tid]]   (inclusive)
     last  = startindex[tid2pos[tid]+1] (exclusive)
  
   tid2pos is a permutation to specify slice of each tid
*/
typedef struct work_array
{
  kaapi_bitmap_t       map;
  uint8_t              tid2pos[KAAPI_MAX_PROCESSOR];
  long                 startindex[1+KAAPI_MAX_PROCESSOR];
} kaapic_work_distribution_t;


/* work container information */
typedef struct work_info
{
  /* grains */
  long par_grain;
  long seq_grain;

} kaapic_work_info_t;


/* local work: used by each worker to process its work */
typedef struct local_work
{
  kaapi_workqueue_t       cr;
#if defined(KAAPIC_USE_KPROC_LOCK)
#else
  kaapi_lock_t            lock;
#endif
  int volatile            init;      /* !=0 iff init */

  void*                   context __attribute__((aligned(KAAPI_CACHE_LINE)));
  struct global_work*     global;    /* go up to all local information */
  kaapi_workqueue_index_t workdone;  /* to compute completion */
  int                     tid;       /* identifier : ==kid */
} kaapic_local_work_t __attribute__ ((aligned (KAAPI_CACHE_LINE)));


/* global work common 
   Initialized by one thread (the master thread).
   Contains all local works and global information to compute completion.
   - wa: the work distribution structure 
   - lwork[tid]; the work for the thread tid
*/
typedef struct global_work
{
  kaapi_atomic64_t workremain __attribute__((aligned(KAAPI_CACHE_LINE)));
  kaapi_atomic32_t workerdone;
  
  /* global distribution */
  kaapic_work_distribution_t wa  __attribute__((aligned(KAAPI_CACHE_LINE)));
  kaapic_local_work_t lwork[KAAPI_MAX_PROCESSOR];

  /* work routine */
  kaapic_foreach_body_t body_f;
  kaapic_body_arg_t*    body_args;

  /* infos container */
  kaapic_work_info_t wi;
} kaapic_global_work_t;



/* Lower level function used by libgomp implementation */

/* init work 
   \retval returns non zero if there is work to do, else returns 0
*/
extern kaapic_local_work_t*  kaapic_foreach_workinit(
  kaapi_thread_context_t*       self_thread,
  kaapi_workqueue_index_t       first, 
  kaapi_workqueue_index_t       last,
  const kaapic_foreach_attr_t*  attr,
  kaapic_foreach_body_t         body_f,
  kaapic_body_arg_t*            body_args
);


/*
*/
extern kaapic_global_work_t* kaapic_foreach_global_workinit
(
  kaapi_thread_context_t* self_thread,
  kaapi_workqueue_index_t first, 
  kaapi_workqueue_index_t last,
  const kaapic_foreach_attr_t*  attr,
  kaapic_foreach_body_t   body_f,
  kaapic_body_arg_t*      body_args
);


/* init local work if know global work.
   May be called by each runing threads that decide to cooperate together
   to execute in common a global work.
   \retval returns non zero if there is work to do, else returns 0
*/
extern kaapic_local_work_t* kaapic_foreach_local_workinit(
  kaapic_local_work_t*    lwork,
  kaapi_workqueue_index_t first,
  kaapi_workqueue_index_t last
);


extern int kaapic_global_work_pop
(
  kaapic_global_work_t* gw,
  kaapi_processor_id_t tid, 
  kaapi_workqueue_index_t* i, 
  kaapi_workqueue_index_t* j
);

/* 
  Return !=0 iff first and last have been filled for the next piece
  of work to execute
*/
extern int kaapic_foreach_worknext(
  kaapic_local_work_t*    work,
  kaapi_workqueue_index_t* first,
  kaapi_workqueue_index_t* last
);


/* To be called by the caller of kaapic_foreach_local_workinit
   that returns success
*/
extern int kaapic_foreach_local_workend(
  kaapi_thread_context_t* self_thread,
  kaapic_local_work_t*    lwork
);


/*
*/
int kaapic_foreach_workend
(
  kaapi_thread_context_t* self_thread,
  kaapic_local_work_t*    work
);

#if defined(__cplusplus)
}
#endif

#endif /* KAAPIC_HIMPL_INCLUDED */
