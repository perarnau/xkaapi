/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** 
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threadctxts.
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

static void kaapi_print_pad(FILE* file, int pad)
{
  for (int i=0; i<pad; ++i)
    fputc(' ', file);
}


/**
*/
static int kaapi_task_descriptor_print( FILE* file, int pad, kaapi_taskdescr_t* td )
{
  kaapi_activationlink_t* lk;
  kaapi_activationlist_t* bcast;

  kaapi_print_pad(file, pad);
  fprintf(file, "td: %p task->%p:", (void*)td, (void*)td->task);

  /* activation list */
  lk = td->list.front;
  if (lk !=0) 
  {
    kaapi_taskdescr_t* tda;
    fprintf(file, " activate: ");
    while (lk !=0)
    {
      tda = lk->td;
      fprintf(file, "(td: %p, wc: %i, task: %p) ", 
            (void*)tda, KAAPI_ATOMIC_READ(&tda->counter), 
            (void*)tda->task);
      lk = lk->next;
    }
  }
  
  /* bcast list */
  bcast = td->bcast;
  if (bcast !=0)
  {
    kaapi_taskdescr_t* tda;
    lk = bcast->front;
    fprintf(file, "bcast:\n");
    while (lk !=0)
    {
      tda = lk->td;
      kaapi_task_descriptor_print(file, 2+pad, tda);
      lk = lk->next;
    }
  }
  fprintf(file, "\n");
  return 0;
}


/**
*/
int kaapi_thread_readylist_print( FILE* file, kaapi_tasklist_t* tl )
{
  kaapi_taskdescr_t* curr = tl->front;
  while (curr != 0)
  {
    fprintf(file, "ready ");
    kaapi_task_descriptor_print(file, 0, curr);
    curr = curr->next;
  }
  return 0;
}

