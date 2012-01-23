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
#include "libgomp.h"


kaapi_libgompctxt_t* GOMP_get_ctxt()
{
  kaapi_processor_t* kproc = kaapi_get_current_processor();
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


int
omp_get_num_threads (void)
{
  return GOMP_get_ctxt()->numthreads;
}

int
omp_get_thread_num (void)
{
  return GOMP_get_ctxt()->threadid;
}


static void GOMP_trampoline_spawn(
  int numthreads,
  int threadid,
  void (*fn) (void *),
  void *data
)
{
  kaapi_libgompctxt_t* ctxt = GOMP_get_ctxt();
  
  ctxt->numthreads = numthreads;
  ctxt->threadid   = threadid;
  fn(data);
}

void 
GOMP_parallel_start (void (*fn) (void *), void *data, unsigned num_threads)
{
  kaapic_begin_parallel();

  if (num_threads == 0)
    num_threads = kaapic_get_concurrency ();
  
  kaapi_libgompctxt_t* ctxt = GOMP_get_ctxt();
  /* do not save the ctxt, assume just one top level ctxt */

  KAAPI_ATOMIC_WRITE(&global_single, 0);
  gomp_barrier_init (&global_barrier, num_threads);


  /* The master thread (id 0) calls fn (data) directly. That's why we
     start this loop from id = 1.*/
  for (int i = 1; i < num_threads; i++)
    kaapic_spawn (4, 
       GOMP_trampoline_spawn,
       KAAPIC_MODE_V, num_threads, 1, KAAPIC_TYPE_INT,
       KAAPIC_MODE_V, i, 1, KAAPIC_TYPE_INT,
       KAAPIC_MODE_V, fn, 1, KAAPIC_TYPE_PTR,
       KAAPIC_MODE_V, data, 1, KAAPIC_TYPE_PTR
    );

  /* initialize master context */
  ctxt->numthreads = num_threads;
  ctxt->threadid = 0;
}

void 
GOMP_parallel_end (void)
{
  kaapic_end_parallel (0); 
  /* implicit sync */
}
