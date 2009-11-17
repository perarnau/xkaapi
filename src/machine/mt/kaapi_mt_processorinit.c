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
#include <unistd.h>
#include <sys/mman.h>

/*
*/
int kaapi_processor_init( kaapi_processor_t* kproc )
{
  size_t k_stacksize;
  size_t k_sizetask;
  size_t k_sizedata;
  size_t pagesize;
  size_t count_pages;
  char* buffer;
  
  kproc->kid          = -1U;
  kproc->hlevel       = 0;
  kproc->hindex       = 0;
  kproc->hlcount      = 0;
  kproc->hkids        = 0;
  
  kaapi_listrequest_init( &kproc->hlrequests );

  KAAPI_STACK_CLEAR( &kproc->lsuspend );

  kproc->fnc_selecarg = 0;
  kproc->fnc_select   = default_param.wsselect;
  
  /* allocate a stack */
  k_stacksize = default_param.stacksize;
  pagesize = getpagesize();
  count_pages = (k_stacksize + pagesize -1 ) / pagesize;
  k_stacksize = count_pages*pagesize;
  buffer = (char*) mmap( 0, k_stacksize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, (off_t)0 );
  if (buffer == (char*)-1) {
    int err __attribute__((unused)) = errno;
    return ENOMEM;
  }
  
  k_sizetask = k_stacksize / 2;
  k_sizedata = k_stacksize - k_sizetask;

  kaapi_stack_init( &kproc->stack, k_sizetask, buffer, k_sizedata, buffer + k_sizetask);
  kproc->stack.requests   = kproc->hlrequests.requests;
  kproc->stack.hasrequest = (volatile int*)&kproc->hlrequests.count;

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
