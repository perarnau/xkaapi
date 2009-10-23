/*
** xkaapi
** 
** Created on Tue Mar 31 15:21:18 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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
#include "kaapi_stealapi.h"

/*
*/
kaapi_steal_processor_t* kaapi_all_stealprocessor[1+KAAPI_MAXSTACK_STEAL];

/*
*/
kaapi_atomic_t kaapi_index_stacksteal = { 0 };

/*
*/
int volatile kaapi_stealapi_term = 0;

/*
*/
kaapi_barrier_td_t kaapi_stealapi_barrier_term;

/**
*/
void stack_destructor(void *k)
{
  if (k ==0) return;
  kaapi_steal_processor_t* kss __attribute__((__unused__)) = (kaapi_steal_processor_t*)k;
  /* free stack of the steal processor context */
}

/**
*/
void kaapi_stealapi_initialize()
{
  int i;
  static int called = 0;
  if (called) return;
  called = 1;

  for (i=0; i<KAAPI_MAXSTACK_STEAL; ++i)
  {
    kaapi_all_stealprocessor[i] = 0;
  }

  /* barrier to detect terminaison */
  kaapi_barrier_td_init( &kaapi_stealapi_barrier_term, 0);

  /* count self thread as stealer */
  kaapi_barrier_td_setactive( &kaapi_stealapi_barrier_term, 1 );
}


/**
*/
void kaapi_stealapi_terminate()
{
  static int called = 0;
  if (called) return;
  called = 1;

  kaapi_stealapi_term = 1;
  kaapi_barrier_td_setactive( &kaapi_stealapi_barrier_term, 0 );

  while (!kaapi_barrier_td_isterminated( &kaapi_stealapi_barrier_term)) kaapi_yield(); 

#if 0 /* seems that kss is already delete (...) */
  kaapi_steal_stack_t* kss = (kaapi_steal_stack_t*)pthread_getspecific(kaapi_current_stack_key);
  kaapi_steal_stack_terminate( kss );
#endif
}



