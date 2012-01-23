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

static void GOMP_trampoline_task(
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

void GOMP_task(
  void (*fn) (void *), 
  void *data, 
  void (*cpyfn) (void *, void *),
  long arg_size, 
  long arg_align, 
  bool if_clause,
  unsigned flags __attribute__((unused))
)
{
  if (!if_clause) 
  {
    if (cpyfn)
	{
	  char buf[arg_size + arg_align - 1];
	  char *arg = (char *) (((uintptr_t) buf + arg_align - 1)
                            & ~(uintptr_t) (arg_align - 1));
	  cpyfn (arg, data);
	  fn (arg);
	}
    else
      fn (data);
  }
  else {
    kaapi_thread_t* thread = kaapi_self_thread();
    void* argtask = kaapi_thread_pushdata_align( thread, arg_size, arg_align);
    if (cpyfn)
      cpyfn(argtask, data);
    else
      memcpy(argtask, data, arg_size);

    kaapi_libgompctxt_t* ctxt = GOMP_get_ctxt();      
    kaapic_spawn( 1,
       GOMP_trampoline_task,
       KAAPIC_MODE_V, ctxt->numthreads, 1, KAAPIC_TYPE_INT,
       KAAPIC_MODE_V, ctxt->threadid, 1, KAAPIC_TYPE_INT,
       KAAPIC_MODE_V, fn, 1, KAAPIC_TYPE_PTR,
       KAAPIC_MODE_V, data, 1, KAAPIC_TYPE_PTR
    );
  }
}

void GOMP_taskwait (void)
{
  kaapic_sync();
}

