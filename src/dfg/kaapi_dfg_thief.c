/*
** xkaapi
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


/** Method call to execute a theft closure.
*/
/**
*/
void kaapi_dfg_thief_entrypoint(kaapi_steal_context_t* stealcontext, void* data)
{
  kaapi_dfg_stack_t s;
  kaapi_dfg_frame_t* top_saved;
#ifdef STACK_LINK_LIST
  kaapi_dfg_frame_t  frame;
  kaapi_dfg_frame_t* f = &frame;
#else
  kaapi_dfg_frame_t* f;
#endif    
  kaapi_dfg_closure_t* work = *(kaapi_dfg_closure_t**)data;
  
#if 0 
  /* Here : should create new version of W access (W, CW or RW) 
     The local thread will compute on the new version and at the execution
     of the terminaison code by the victim, the victim will report new version of the data to its
     control flow.
     Question: where to allocate the new version? We apply the owner compute rule.... with heap allocation
  */
  /* Reified the closure if not */
  if (!KAAPI_DFG_CLOSURE_ISREIFIED( work )) 
  {
    kaapi_dfg_reify_closure( work );
  }
#endif


  KAAPI_DFG_STACK_INIT_WITH_SC( &s, stealcontext );
  KAAPI_DFG_FRAME_INIT(f);
  KAAPI_DFG_STACK_PUSH( &s, top_saved, f );
  /* execute the closure */
  KAAPI_CLOSURE_EXECUTE_FROM_THIEF( &s, &f, work);

  int** sres = (int**) KAAPI_FORMAT_GET_SHARED(KAAPI_DFG_CLOSURE_GETFORMAT(work), work, 1);
  
  /* mark it as terminated */
  KAAPI_DFG_CLOSURE_SETSTATE( work, KAAPI_CLOSURE_TERM );
}

/**
*/
void kaapi_dfg_thiefcode(
  kaapi_dfg_stack_t* s, kaapi_dfg_closure_t* c 
)
{
}
