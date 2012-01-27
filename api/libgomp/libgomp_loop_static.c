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
#include <stdio.h>

#include "kaapic_impl.h"

/*
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
*/

static void gomp_foreach_wrapper(
  int32_t i, int32_t j, int32_t tid, 
  kaapic_body_arg_t* call
)
{
//TODO
}

/* */
bool GOMP_loop_static_start (
  long start, long end, long incr, 
  long chunk_size, 
  long *istart, 
  long *iend
)
{
#if 0
  kaapi_thread_t* thread = kaapi_self_thread();
  kaapic_foreach_attr_t attr;
  kaapic_foreach_attr_init(&attr);
  kaapic_foreach_attr_set_grains(&attr, chunk_size, chunk_size);
  kaapic_work_t* work = kaapi_thread_pushdata(thread, sizeof(kaapic_work_t));
  kaapic_body_arg_t* fnc_arg = kaapi_thread_pushdata(thread, 
      offsetof(kaapic_body_arg_t, args)+4*sizeof(void*)
  );
//TODO  fnc_arg->u.f_c   = gomp_foreach_wrapper;
  fnc_arg->nargs   = 4;
  fnc_arg->args[0] = 0; /* where is the entry point... not needed */
  fnc_arg->args[1] = 0;
  fnc_arg->args[2] = (void*)start;
  fnc_arg->args[3] = (void*)incr;

  /* normalize iteration: to 0,M */
  kaapic_foreach_workinit( work, 0, (end-start)/incr, attr, gomp_foreach_wrapper, fnc_arg );

  printf("%s:: \n", __FUNCTION__);
#endif

  return (bool)0;
}

bool GOMP_loop_static_next (long *istart, long *iend)
{
  /* seems never called except through runtime selection ? */
  printf("%s:: \n", __FUNCTION__);

  return (bool)0;
}


void GOMP_parallel_loop_static_start(
    void (*fn) (void *), 
    void *data,
    unsigned num_threads, 
    long start, long end, long incr, long chunk_size
)
{
  /* seems never called except through runtime selection ? */
  printf("%s:: \n", __FUNCTION__);
}

