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

/**
 */
static const char* tab_state[] = {
  "CREATED",         /* created */
  "PARTITIONNING",   /* paritionning scheme beguns */
  "PARTITIONNED",    /* multi partition ok */
  "EXECUTING",       /* exec state started */
  "WAIT"             /* end of execution */
};


/**
 */
__attribute__((unused))
static void kaapi_threadgroup_printdata
(FILE* file, int tid, kaapi_hashmap_t* hm )
{
#if 0
#warning "TODO"
  kaapi_hashentries_t* entry;
  
  for (uint32_t i=0; i<KAAPI_HASHMAP_SIZE; ++i)
  {
    entry = get_hashmap_entry( hm, i );
    while (entry !=0) 
    {
      kaapi_version_t* ver = entry->u.version;
      kaapi_part_datainfo_t* verinfo = kaapi_version_reader_find_tid( ver, tid );
      if (verinfo !=0)
      {
        if (verinfo->reader.used) {
          fprintf(file, "@:%p -> local @:%p\n", ver->original_data, verinfo->reader.addr );
        }
        if (verinfo->delete_data !=0)
          fprintf(file, "@:%p -> to delete @:%p\n", ver->original_data, verinfo->delete_data );
      }
      entry = entry->next;
    }
  }
#endif
}



#if 0
/**
 */
static void kaapi_threadgroup_printinputoutputdata(
  FILE* file, 
  int tid, 
  kaapi_hashmap_t* hm, 
  kaapi_vector_t* v 
)
{
  kaapi_vectentries_bloc_t* entry;
  
  if (v->firstbloc ==0) return;
  
  entry = v->firstbloc;
  while (entry !=0) 
  {
    for (size_t i=0; i<entry->pos; ++i)
    {
      kaapi_pidreader_t* reader = &entry->data[i];
      fprintf(file, "@:%p -> first reader on tid: %i by task @:%p, param:%i\n", 
             reader->addr, 
             reader->tid, 
             (void*)reader->task, 
             (int)reader->used 
             );
    }
    entry = entry->next;
  }
  
  entry = v->firstbloc;
  while (entry !=0) 
  {
    for (size_t i=0; i<entry->pos; ++i)
    {
      kaapi_pidreader_t* reader   = &entry->data[i];
      kaapi_hashentries_t* hentry = kaapi_hashmap_find(hm, reader->addr);
      if (hentry ==0) {
        printf("*** error, cannot find data with address:%p\n", reader->addr);
      }
      else {
        kaapi_version_t* ver = hentry->u.version;
        printf("@:%p -> last writer on tid:%i by task @:%p\n", 
               reader->addr, 
               ver->writer_thread,
               (void*)ver->writer.task 
               );
      }
    }
    entry = entry->next;
  }
}
#endif




/**
*/
static int kaapi_thread_recv_comlist_print( FILE* file, kaapi_comlink_t* cl )
{
  kaapi_activationlink_t* al;
  while (cl != 0)
  {
    fprintf(file, "[rsignal: %p, tag:%llu tid:%i laddr:%p lsize:(%lu x %lu) -> ",
            (void*)cl->u.recv,
            cl->u.recv->tag,
            cl->u.recv->tid,
            cl->u.recv->data, 
            cl->u.recv->view.size[0],
            cl->u.recv->view.size[1]
    );

    al = cl->u.recv->list.front;
    while (al !=0)
    {
      fprintf(file, "(td:%p, wc:%i, task:%p) ", 
          (void*)al->td, 
          KAAPI_ATOMIC_READ(&al->td->counter),
          (void*)al->td->task
      );
      al = al->next;
    }
    fprintf(file, ")\n");
    cl = cl->next;
  }
  return 0;
}

/**
*/
static int kaapi_thread_send_comaddr_print( FILE* file, kaapi_comsend_raddr_t* lraddr )
{
  while (lraddr !=0)
  {
    fprintf(file, "(tag:%u, asid:", (unsigned)lraddr->tag);
    kaapi_memory_address_space_fprintf( file, lraddr->asid );
    fprintf(file,", rsignal:%p,  raddr:%p, rsize:(%lu x %lu) ) ", 
        (void*)lraddr->rsignal, 
        (void*)lraddr->raddr, 
        lraddr->rview.size[0], 
        lraddr->rview.size[1] 
    );
    lraddr = lraddr->next;
  }
  return 0;
}

/**
*/
static int kaapi_thread_send_comlist_print( FILE* file, kaapi_comlink_t* cl )
{
  kaapi_comsend_raddr_t* lraddr;
  while (cl != 0)
  {
    fprintf(file, "[datatag:%u; laddr:%p lsize:(%lu x %lu) ->", 
        (unsigned)cl->u.send->vertag, 
        cl->u.send->data, 
        cl->u.send->view.size[0],
        cl->u.send->view.size[1]
    );
    lraddr = &cl->u.send->front;
    kaapi_thread_send_comaddr_print( file, lraddr );
    fprintf(file, "]\n");
    cl = cl->next;
  }
  return 0;
}

/**
*/
static int kaapi_task_descriptor_print( FILE* file, kaapi_taskdescr_t* td )
{
  kaapi_activationlink_t* lk;
  kaapi_taskbcast_arg_t* bcast;
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
      fprintf(file, "(td: %p, wc: %i, task: %p) ", (void*)tda, KAAPI_ATOMIC_READ(&tda->counter), (void*)tda->task);
      lk = lk->next;
    }
  }
  
  /* bcast list */
  bcast = td->bcast;
  if (bcast !=0)
  {
    kaapi_comsend_t* com = &bcast->front;
    fprintf(file, " bcast:");
    while (com !=0)
    {
      kaapi_comsend_raddr_t* lraddr = &com->front;
      fprintf(file, "\n\t[laddr:%p, lsize:(%lu x %lu) ->", 
          com->data, 
          com->view.size[0], 
          com->view.size[1]  
      );
      kaapi_thread_send_comaddr_print( file, lraddr );
      fprintf(file, "]\n");
      com = com->next;
    }
  }
  fprintf(file, "\n");
  return 0;
}


#if 0
/**
*/
int kaapi_thread_readylist_print( FILE* file, kaapi_tasklist_t* tl )
{
  kaapi_taskdescr_t* curr = tl->front;
  while (curr != 0)
  {
    fprintf(file, "ready ");
    kaapi_task_descriptor_print(file, curr);
    curr = curr->next;
  }
  return 0;
}
#endif


/**
 */
int kaapi_threadgroup_print( FILE* file, kaapi_threadgroup_t thgrp )
{
  int i;
  if (thgrp ==0) return EINVAL;
  if (file ==0) return EINVAL;
  fprintf(file, "ThreadGroup size=%i, state=%s\n", thgrp->group_size, tab_state[thgrp->state] );
    
  for (i=-1; i<thgrp->group_size; ++i)
  {
    if (thgrp->localgid == thgrp->tid2gid[i])
    {
      fprintf(file, "\n\nPartition %i/%i, on gid:%u\n", i, thgrp->group_size, thgrp->localgid);
      kaapi_thread_print( file, thgrp->threadctxts[i] );
      fprintf(file, "\n*** Send on thread %i/%i:\n", i, thgrp->group_size);
      kaapi_thread_send_comlist_print( file, thgrp->lists_send[i] );
      fprintf(file, "\n*** Recv on thread %i/%i:\n", i, thgrp->group_size);
      kaapi_thread_recv_comlist_print( file, thgrp->lists_recv[i] );
      fflush(file);
    }
  }
#if 0
  fprintf(file, "First input data on thread %i/%i:\n", i, thgrp->group_size);
  kaapi_threadgroup_printinputoutputdata(file, i, &thgrp->ws_khm, &thgrp->ws_vect_input );
#endif
  
  return 0;  
}
