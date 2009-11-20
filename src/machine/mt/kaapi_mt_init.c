/*
** kaapi_init.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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
#include <stdlib.h>

/*
*/
kaapi_uint32_t kaapi_count_kprocessors = 0;

/*
*/
kaapi_processor_t** kaapi_all_kprocessors = 0;


/*
*/
pthread_key_t kaapi_current_processor_key;

/*
*/
pthread_key_t c;

/* 
*/
kaapi_atomic_t kaapi_term_barrier = { 0 };

/* 
*/
volatile int kaapi_isterm = 0;

/** Should be with the same file as kaapi_init
 */
void _kaapi_dummy(void* foo)
{
}

/** Dependencies with kaapi_stack_t* kaapi_self_stack(void)
*/
kaapi_stack_t* kaapi_self_stack(void)
{
  return _kaapi_self_stack();
}

/**
*/
void __attribute__ ((constructor)) kaapi_init(void)
{
  kaapi_isterm = 0;
  
  /* set up runtime parameters */
  kaapi_setup_param(0,0);
  
  /* initialize the kprocessor key */
  kaapi_assert( 0 == pthread_key_create( &kaapi_current_processor_key, 0 ) );
    
  /* setup topology information */
  kaapi_setup_topology();

  /* set the kprocessor AFTER topology !!! */
  kaapi_assert_m( 0, kaapi_setconcurrency( default_param.cpucount ), "kaapi_setconcurrency" );
  
  pthread_setspecific( kaapi_current_processor_key, kaapi_all_kprocessors[0] );

  /* dump output information */
  printf("[KAAPI::INIT] use #physical cpu:%u\n", default_param.cpucount);
}


/**
*/
void __attribute__ ((destructor)) kaapi_fini(void)
{
  int i;
  
  printf("[KAAPI::TERM]\n");
  fflush( stdout );

  /* wait end of the initialization */
  kaapi_isterm = 1;
  kaapi_barrier_td_setactive(&kaapi_term_barrier, 0);
  
  while (!kaapi_barrier_td_isterminated( &kaapi_term_barrier ))
  {
    kaapi_sched_advance( kaapi_all_kprocessors[0] );
  }
  
  for (i=0; i<kaapi_count_kprocessors; ++i)
  {
    free(kaapi_all_kprocessors[i]);
    kaapi_all_kprocessors[i]= 0;
  }
  free( kaapi_all_kprocessors );

  kaapi_all_kprocessors =0;
  
  /* TODO: destroy topology data structure */
}
