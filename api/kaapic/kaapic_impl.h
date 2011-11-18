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

#include "kaapi.h"
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
  void*               args[1];
} kaapic_body_arg_t;

/* Signature of foreach body 
   Called with (first, last, tid, arg) in order to do
   computation over the range [first,last[.
*/
typedef void (*kaapic_foreach_body_t)(int32_t, int32_t, int32_t, void* );


/* exported foreach interface 
   evaluate body_f(first, last, body_args) in parallel, assuming
   that the evaluation of body_f(i, j, body_args) does not impose
   dependency with evaluation of body_f(k,l, body_args) if [i,j[ and [k,l[
   does not intersect.
*/
extern int kaapic_foreach_common
(
  int32_t               first, 
  int32_t               last,
  kaapic_foreach_body_t body_f,
  void*                 body_args
);

/* wrapper for kaapic_foreach(...) and kaapic_foreach_withformat(...)
*/
extern void kaapic_foreach_body2user(int32_t first, int32_t last, int32_t tid, void* arg );


/*
*/
typedef struct arg_info_t
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

} arg_info_t;

/*
*/
typedef struct task_info
{
  kaapic_spawn_fn_t body;
  uintptr_t         nargs;
  arg_info_t        args[1];
} task_info_t;

extern int kaapic_spawn_ti(
  kaapi_thread_t* thread, 
  kaapi_task_body_t body, 
  task_info_t* ti
);

#if defined(__cplusplus)
}
#endif

#endif /* KAAPIC_HIMPL_INCLUDED */
