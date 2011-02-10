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
  const char* name = 0;
  const kaapi_format_t* fmt;
  kaapi_task_body_t body;

  body = kaapi_task_getbody(td->task);
  fmt = kaapi_format_resolvebybody( body );
  if (fmt !=0) 
    name = fmt->name;
  else {
    if (body == kaapi_taskmove_body)
      name = "move";
    else 
      name = "<undef>";
  }
    

  kaapi_print_pad(file, pad);
  fprintf(file, "td:%p  date:%lu  task->%p  name:%s", 
    (void*)td, td->date, (void*)td->task,
    name
  );

  /* activation list */
  lk = td->list.front;
  if (lk !=0) 
  {
    kaapi_taskdescr_t* tda;
    fprintf(file, "\n\tactivate: ");
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
    fprintf(file, "\n\tbcast:\n");
    while (lk !=0)
    {
      tda = lk->td;
      kaapi_task_descriptor_print(file, 4+pad, tda);
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
  /* new history of visited data */
  kaapi_hashmap_t visit_khm;
  kaapi_hashentries_t* entry;
  kaapi_activationlink_t* lk;

  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &visit_khm, 0 );

  fprintf(file, "*** ready task:\n");
  kaapi_taskdescr_t* td = tl->front;
  while (td != 0)
  {
    entry = kaapi_hashmap_findinsert(&visit_khm, td);
    if (entry->u.data.tag ==0)
    { /* first time I visit it: print and insert activated task descr into the hashmap */
      kaapi_task_descriptor_print(file, 0, td);
      entry->u.data.tag = 1 + td->date; 
      entry->u.data.ptr = td; 

      /* add other td */
      lk = td->list.front;
      if (lk !=0) 
      {
        kaapi_taskdescr_t* tda;
        while (lk !=0)
        {
          tda = lk->td;
          entry = kaapi_hashmap_findinsert(&visit_khm, tda);
          entry->u.data.ptr = tda; 
          lk = lk->next;
        }
      }
    }
    td = td->next;
  }
  
  /* now print all non printed task descriptor */
  fprintf(file, "*** non ready task:\n");
  for (int i=0; i<KAAPI_HASHMAP_SIZE; ++i)
  {
    entry = _get_hashmap_entry(&visit_khm, i);
    while (entry != 0)
    {
      if (entry->u.data.tag ==0)
        kaapi_task_descriptor_print(file, 0, (kaapi_taskdescr_t*)entry->u.data.ptr);
      entry = entry->next;
    }
  }
  
  kaapi_hashmap_destroy(&visit_khm);
  return 0;
}

