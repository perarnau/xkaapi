/*
** kaapi_dfg.c
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
#include "kaapi_impl.h"
#include <execinfo.h>


static int kaapi_dfg_closure_is_term(void *arg)
{
  kaapi_dfg_closure_t* clo = (kaapi_dfg_closure_t*)arg;
  
  return KAAPI_DFG_CLOSURE_GETSTATE(clo) == KAAPI_CLOSURE_TERM;
}


/** 
*/
void kaapi_dfg_reify_closure( kaapi_dfg_closure_t* c )
{
  if (KAAPI_DFG_CLOSURE_ISREIFIED(c)) return;
  if (KAAPI_DFG_CLOSURE_GETFORMAT(c)->reificator !=0)
  {
    (*KAAPI_DFG_CLOSURE_GETFORMAT(c)->reificator)(c);
  }
  KAAPI_DFG_CLOSURE_SETREIFIED(c);
}



/**
*/
int kaapi_save_dfg( int savemode, int n )
{
  void* callstack[128];
  int i, frames = backtrace(callstack, 128);
  char** strs = backtrace_symbols(callstack, frames);
  for (i = 0; i < frames; ++i) {
     printf("%s\n", strs[i]);
  }
  free(strs);
  return n;
}
