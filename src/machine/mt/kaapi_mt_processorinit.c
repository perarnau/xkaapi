/*
** kaapi_mt_processorinit.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
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
#include "kaapi_impl.h"

/*
*/
int kaapi_processor_init( kaapi_processor_t* kproc )
{
  kaapi_thread_context_t* ctxt;
  size_t k_stacksize;
  size_t k_sizetask;
  size_t k_sizedata;

  kproc->thread       = 0;  
  kproc->kid          = -1U;
  kproc->issteal      = 0;
  kproc->hlevel       = 0;
  kproc->hindex       = 0;
  kproc->hlcount      = 0;
  kproc->hkids        = 0;
  
  KAAPI_ATOMIC_WRITE( &kproc->lock, 0);
  
  kaapi_listrequest_init( kproc, &kproc->hlrequests );

  kaapi_wsqueuectxt_init( &kproc->lsuspend );
  kproc->readythread = 0;
  kaapi_sched_initready(kproc);
  kaapi_lfree_clear( kproc );

  kproc->fnc_selecarg = 0;
  kproc->fnc_select   = kaapi_default_param.wsselect;
  

  /* workload */
  kproc->workload._counter= 0;
  
  /* allocate a stack */
  k_stacksize = kaapi_default_param.stacksize;
  k_sizetask  = k_stacksize / 2;
  k_sizedata  = k_stacksize - k_sizetask;

  ctxt = (kaapi_thread_context_t*)kaapi_context_alloc( kproc );
  /* set new context to the kprocessor */
  kaapi_setcontext(kproc, ctxt);
  
  return 0;
}


int kaapi_processor_setuphierarchy( kaapi_processor_t* kproc )
{
  int i;
  kproc->hlevel    = 1;
  kproc->hindex    = calloc( kproc->hlevel, sizeof(kaapi_uint16_t) );
  kproc->hlcount   = calloc( kproc->hlevel, sizeof(kaapi_uint16_t) );
  kproc->hkids     = calloc( kproc->hlevel, sizeof(kaapi_processor_id_t*) );
/*  for (i=0; i<kproc->hlevel; ++i) */
  {
    kproc->hindex[0]  = kproc->kid; /* only one level !!!! */
    kproc->hlcount[0] = kaapi_count_kprocessors;
    kproc->hkids[0]   = calloc( kproc->hlcount[0], sizeof(kaapi_processor_id_t) );
    for (i=0; i<kproc->hlcount[0]; ++i)
      kproc->hkids[0][i] = i;  
  }  
      
  return 0;
}
