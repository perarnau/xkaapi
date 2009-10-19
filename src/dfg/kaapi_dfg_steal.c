/*
** ckaapi
** 
** Created on Tue Mar 31 15:22:28 2009
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
#include "kaapi_dfg.h"


/** 
*/
#if 0
void kaapi_dfg_reify_closure( kaapi_dfg_closure_t* c )
{
  if (KAAPI_DFG_CLOSURE_ISREIFIED(c)) return;
  if (KAAPI_DFG_CLOSURE_GETFORMAT(c)->reificator !=0)
  {
    (*KAAPI_DFG_CLOSURE_GETFORMAT(c)->reificator)(c);
  }
  KAAPI_DFG_CLOSURE_SETREIFIED(c);
}
#endif


/**
*/
void kaapi_dfg_update_access( kaapi_access_t* a, kaapi_dfg_closure_t* clo, int i )
{
  ckaapi_assert_debug( KAAPI_FORMAT_GET_MODE(KAAPI_DFG_CLOSURE_GETFORMAT(clo), i) != KAAPI_ACCESS_MODE_V );
  if (KAAPI_DFG_ACCESS_NEW_VERSION(a)) 
  {
    /* test if user shared points to the same data or not */
    if (*KAAPI_FORMAT_GET_SHARED(KAAPI_DFG_CLOSURE_GETFORMAT(clo), clo, i)  != a->_version) 
    {
      /* delete previous value *a->_user_shared 
         could do that because 1/ the current closure had produce a new version, it is terminated 
         and this is the owner thread that execute the closure :
          - all previous closure has been finished, thus it does not exist concurrent closure that
          access to the data.
          - next closures may been theft by using heavy representation and thus, making forward te pointer
          of the version to update to date version.
      */
      /* assume *a->_user_shared is on stack */
      *KAAPI_FORMAT_GET_SHARED(KAAPI_DFG_CLOSURE_GETFORMAT(clo),clo,i) = a->_version;
    }
  }
}


/** 
*/
int kaapi_dfg_closure_isready( kaapi_dfg_closure_t* c )
{
  int i;
  kaapi_dfg_format_closure_t* fmt = KAAPI_DFG_CLOSURE_GETFORMAT(c);
  int isready = 1;

#if 0
  size_t sz_param = 0;
  size_t sz_access = 0;
#endif

  if (KAAPI_DFG_CLOSURE_GETSTATE(c) != KAAPI_CLOSURE_INIT) return 0;

  for (i=0; (i<fmt->count_params) && (isready !=0); ++i)
  {
    kaapi_access_mode_t m = KAAPI_FORMAT_GET_MODE(fmt, i);
#if 0
    sz_param += KAAPI_FORMAT_GET_SIZE(fmt, i);
#endif
    if (m == KAAPI_ACCESS_MODE_V) continue;
#if 0
    sz_access += KAAPI_FORMAT_GET_SIZE(fmt, i);
#endif
    if (KAAPI_ACCESS_IS_ONLYWRITE(m) || KAAPI_ACCESS_IS_POSTPONED(m)) continue;
    if (!KAAPI_DFG_CLOSURE_ISREIFIED( c)) 
      kaapi_dfg_reify_closure( c );

    /* R or RW */
    kaapi_access_t* a = KAAPI_FORMAT_GET_ACCESS(fmt, c, i);
    isready = KAAPI_DFG_ACCESS_ISREADY( a, m );
  }
#if 0
  if (i == fmt->count_params) 
  {
    fmt->size_param = sz_param;
    fmt->size_allaccess = sz_access;
  }
#endif
  return isready;
}


/**
*/
int kaapi_dfg_compute_readyness( kaapi_dfg_stack_t* s, kaapi_dfg_closure_t* c )
{
  /* update readyness of access for the closure c */
  return kaapi_dfg_closure_isready(c);
}


/** Called by the victim to try to steal work.
*/
void kaapi_dfg_stealercode( 
  kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request
)
{
  kaapi_dfg_stack_t* s   = 0;
  kaapi_dfg_closure_t* c = 0;
  int lastrequest = 0;
  /* get the base frame of the stealcontex */
  kaapi_dfg_frame_t* curr = KAAPI_DFG_STACK_BACK(s);
  
#if 0
  printf("\n\nBEGIN DUMP STACK\n");
#endif
  while (curr !=0)
  {
    kaapi_dfg_closure_t* clo;
    KAAPI_FIFO_TOP(curr, clo);
    while (clo !=0)
    {
#if 0
      if (clo == c) 
      {
        clo = clo->_next; 
        continue; 
      }
#endif
      if ( (clo != c) && kaapi_dfg_closure_isready( clo ))
      {
        kaapi_dfg_format_closure_t* fmt = KAAPI_DFG_CLOSURE_GETFORMAT(clo);
#if 0
        printf("Closure: @=0x%x, name:%s is ready\n", clo, fmt->name );
#endif
        KAAPI_DFG_CLOSURE_SETSTATE(clo, KAAPI_CLOSURE_STEAL);

        /* reply to the stealer: pass the closure as parameter */
        while (request[lastrequest] ==0) ++lastrequest;
        kaapi_dfg_closure_t** ptwork;
        if (kaapi_steal_processor_alloc_data( stealcontext, request[lastrequest], 
                                              (void**)&ptwork, sizeof(kaapi_dfg_closure_t*)) ==0)
        {
          *ptwork = clo;
          kaapi_thief_reply_request( request, lastrequest, 1 );
          /*
          kaapi_request_reply( request[lastrequest], stealcontext, &kaapi_dfg_thief_entrypoint, 1, KAAPI_MASTER_FINALIZE_FLAG);          
*/
          --count;
          if (count ==0) return;
        }
        else {
          do {
            kaapi_thief_reply_request( request, lastrequest, 0 );
            --count;
            if (count ==0) return;
            while (request[lastrequest] ==0) ++lastrequest;
          } while (count>0);
        }
      }
#if 0
      else
        if ((KAAPI_DFG_CLOSURE_GETSTATE(clo) == KAAPI_CLOSURE_EXEC) || (clo ==c))
          printf("Closure: @=0x%x, name:%s is not ready: under execution\n", clo, KAAPI_DFG_CLOSURE_GETFORMAT(clo)->name );
        else if (KAAPI_DFG_CLOSURE_GETSTATE(clo) == KAAPI_CLOSURE_STEAL) 
          printf("Closure: @=0x%x, name:%s is not ready: theft\n", clo, KAAPI_DFG_CLOSURE_GETFORMAT(clo)->name );
        else
          printf("Closure: @=0x%x, name:%s is not ready: state=%i\n", clo, KAAPI_DFG_CLOSURE_GETFORMAT(clo)->name, clo->_state );
#endif

#if 0
      /* clo not ready, reify it and update ... */
      if (s->_dfgaccess ==0) {
        s->_dfgaccess = malloc( sizeof(kaapi_map_table_t) );
        kaapi_map_init( s->_dfgaccess );
      }
#endif
      if (!KAAPI_DFG_CLOSURE_ISREIFIED( clo)) 
      {
        kaapi_dfg_reify_closure( clo );
      }
      kaapi_dfg_compute_readyness( s, clo );
      
      clo = clo->_next;
    }
    curr = KAAPI_DFG_STACK_NEXT_FRAME(s, curr);
  }
#if 0
  printf("END DUMP STACK\n\n");
#endif
}
