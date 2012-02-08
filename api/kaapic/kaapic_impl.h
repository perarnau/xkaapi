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
  int32_t                first, 
  int32_t                last,
  kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_t  body_f,
  kaapic_body_arg_t*     body_args
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

extern int kaapic_spawn_ti(
  kaapi_thread_t* thread, 
  kaapi_task_body_t body, 
  kaapic_task_info_t* ti
);


/* work array. allow for random access. */
typedef struct work_array
{
  kaapi_bitmap_value_t map;
  long off;
  long scale;
} work_array_t;


/* work container */
typedef struct work_info
{
#if CONFIG_MAX_TID
  /* maximum thread index */
  unsigned int max_tid;
#endif

  /* grains */
  long par_grain;
  long seq_grain;

} work_info_t;


/* master work */
typedef struct work
{
  kaapi_workqueue_t cr __attribute__((aligned(64)));
#if defined(USE_KPROC_LOCK)
#else
  kaapi_lock_t      lock;
#endif

#if CONFIG_TERM_COUNTER
  /* global work counter */
  kaapi_atomic_t* counter;
#endif

  /* work routine */
  kaapic_foreach_body_t body_f;
  kaapic_body_arg_t*    body_args;

  /* points to next _wa and _wi fields */
  work_array_t* wa;
  work_info_t* wi;

  /* split_root_task container */
  work_array_t _wa;

  /* infos container */
  work_info_t _wi;
  
  void* context; /* return by begin_adapt */

  /* thread context to restore */
  kaapi_frame_t frame;

} kaapic_work_t;


/* thief work: keep reference to initial work array and wi 
   This structure is the same as kaapic_work_t except 
   the container fields _wa and _wi
*/
typedef struct thief_work
{
  kaapi_workqueue_t cr __attribute__((aligned(64)));
#if defined(USE_KPROC_LOCK)
#else
  kaapi_lock_t      lock;
#endif

#if CONFIG_TERM_COUNTER
  /* global work counter */
  kaapi_atomic_t* counter;
#endif

  /* work routine */
  kaapic_foreach_body_t body_f;
  kaapic_body_arg_t*    body_args;


  /* split_root_task */
  work_array_t* wa;

  /* infos */
  const work_info_t* wi;
  void* context_adapt;
  void* user_data[4];

} kaapic_thief_work_t;



/* Lower level function used by libgomp implementation */

/* init work 
   \retval returns non zero if there is work to do, else returns 0
*/
extern int kaapic_foreach_workinit
(
  kaapi_thread_context_t*       self_thread,
  kaapic_work_t*                work,
  kaapi_workqueue_index_t       first, 
  kaapi_workqueue_index_t       last,
  const kaapic_foreach_attr_t*  attr,
  kaapic_foreach_body_t         body_f,
  kaapic_body_arg_t*            body_args
);




/* 
  Return !=0 iff first and last have been filled for the next piece
  of work to execute
*/
extern int kaapic_foreach_worknext(
  kaapic_work_t*           work,
  kaapi_workqueue_index_t* first,
  kaapi_workqueue_index_t* last
);


/*
*/
int kaapic_foreach_workend
(
  kaapi_thread_context_t* self_thread,
  kaapic_work_t*          work
);

#if defined(__cplusplus)
}
#endif

#endif /* KAAPIC_HIMPL_INCLUDED */
