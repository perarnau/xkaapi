/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include "libgomp.h"

/* General comment on: http://gcc.gnu.org/onlinedocs/libgomp/Implementing-FOR-construct.html

4.11 Implementing FOR construct

       #pragma omp parallel for
       for (i = lb; i <= ub; i++)
         body;
becomes

       void subfunction (void *data)
       {
         long _s0, _e0;
         while (GOMP_loop_static_next (&_s0, &_e0))
         {
           long _e1 = _e0, i;
           for (i = _s0; i < _e1; i++)
             body;
         }
         GOMP_loop_end_nowait ();
       }
     
       GOMP_parallel_loop_static (subfunction, NULL, 0, lb, ub+1, 1, 0);
       subfunction (NULL);
       GOMP_parallel_end ();
       
       
       #pragma omp for schedule(runtime)
       for (i = 0; i < n; i++)
         body;
becomes

       {
         long i, _s0, _e0;
         if (GOMP_loop_runtime_start (0, n, 1, &_s0, &_e0))
           do {
             long _e1 = _e0;
             for (i = _s0, i < _e0; i++)
               body;
           } while (GOMP_loop_runtime_next (&_s0, _&e0));
         GOMP_loop_end ();
       }
Note that while it looks like there is trickiness to propagating a non-constant STEP, there isn't really. We're explicitly allowed to evaluate it as many times as we want, and any variables involved should automatically be handled as PRIVATE or SHARED like any other variables. So the expression should remain evaluable in the subfunction. We can also pull it into a local variable if we like, but since its supposed to remain unchanged, we can also not if we like.

If we have SCHEDULE(STATIC), and no ORDERED, then we ought to be able to get away with no work-sharing context at all, since we can simply perform the arithmetic directly in each thread to divide up the iterations. Which would mean that we wouldn't need to call any of these routines.

There are separate routines for handling loops with an ORDERED clause. Bookkeeping for that is non-trivial...

*/
bool GOMP_loop_dynamic_start (
  long start, 
  long end, 
  long incr, 
  long chunk_size,
	long *istart, 
  long *iend
)
{
  int retval;
  
  printf("%s:: \n", __FUNCTION__);
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_thread_context_t* const self_thread = kproc->thread;
  kaapi_libgompctxt_t* ctxt = GOMP_get_ctxtkproc( kproc );
  kaapi_libgompworkshared_t* workshare = &ctxt->workshare;

  kaapi_atomic_lock( &ctxt->teaminfo->lock );

  if (1 /* is master init ? 1 -> yes I'm a slave*/)
  {
    /* I'm a slave with respect to the master */
    /* - steal from master my initial range or a part... */
    /* - initialize my local work with this initial range */
    /* - update my work count */

    retval = 0;
  }
  else {
    /* initialize the master if not already done */
    retval = kaapic_foreach_workinit(self_thread, 
          &ctxt->workshare.work, 
          start, 
          end, 
          0,    /* attr */
          0,    /* body */
          0     /* arg */
      );
  }
  kaapi_atomic_unlock( &ctxt->teaminfo->lock );

  if (retval ==0) return 0;
  
  return kaapic_foreach_worknext(
        &ctxt->workshare.work, 
        istart,
        iend
  );
}

bool GOMP_loop_dynamic_next (long *istart, long *iend)
{
  printf("%s:: \n", __FUNCTION__);

  kaapi_processor_t*   kproc = kaapi_get_current_processor();
  kaapi_libgompctxt_t* ctxt  = GOMP_get_ctxtkproc( kproc );

  return kaapic_foreach_worknext(
        &ctxt->workshare.work, 
        istart,
        iend
  );
}

void GOMP_parallel_loop_dynamic_start (
          void (*fn) (void *), 
          void *data,
				  unsigned num_threads, 
          long start, 
          long end, 
          long incr, 
          long chunk_size
)
{
  printf("%s:: \n", __FUNCTION__);
}

bool GOMP_loop_ordered_dynamic_start (
          long start, 
          long end, 
          long incr,
          long chunk_size, 
          long *istart, 
          long *iend
)
{
  printf("%s:: \n", __FUNCTION__);
}

bool GOMP_loop_ordered_dynamic_next (long *istart, long *iend)
{
  printf("%s:: \n", __FUNCTION__);
}

