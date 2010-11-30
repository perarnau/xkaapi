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
static void kaapi_threadgroup_printdata(FILE* file, int tid, kaapi_hashmap_t* hm )
{
  kaapi_hashentries_t* entry;
  
  for (uint32_t i=0; i<KAAPI_HASHMAP_SIZE; ++i)
  {
    entry = get_hashmap_entry( hm, i );
    while (entry !=0) 
    {
      kaapi_version_t* ver = entry->u.dfginfo;
      if (ver->readers[tid].used) {
        fprintf(file, "@:%p -> local @:%p\n", ver->original_data, ver->readers[tid].addr );
      }
      if (ver->delete_data[tid] !=0)
        fprintf(file, "@:%p -> to delete @:%p\n", ver->original_data, ver->delete_data[tid] );
      entry = entry->next;
    }
  }
}



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
    for (int i=0; i<entry->pos; ++i)
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
    for (int i=0; i<entry->pos; ++i)
    {
      kaapi_pidreader_t* reader   = &entry->data[i];
      kaapi_hashentries_t* hentry = kaapi_hashmap_find(hm, reader->addr);
      if (hentry ==0) {
        printf("*** error, cannot find data with address:%p\n", reader->addr);
      }
      else {
        kaapi_version_t* ver = hentry->u.dfginfo;
        printf("@:%p -> last writer on tid:%i by task @:%p\n", 
               reader->addr, 
               ver->writer_thread,
               (void*)ver->writer_task 
               );
      }
    }
    entry = entry->next;
  }
}


/**
 */
int kaapi_threadgroup_print( FILE* file, kaapi_threadgroup_t thgrp )
{
  int i;
  if (thgrp ==0) return EINVAL;
  if (file ==0) return EINVAL;
  fprintf(file, "ThreadGroup size=%i, state=%s\n", thgrp->group_size, tab_state[thgrp->state] );
  
  fprintf(file, "Main thread\n");
  kaapi_thread_print( file, thgrp->mainctxt );
  
  for (i=0; i<thgrp->group_size; ++i)
  {
    fprintf(file, "Partition %i/%i:\n", i, thgrp->group_size);
    kaapi_thread_print( file, thgrp->threadctxts[i] );
  }
  fprintf(file, "First input data on thread %i/%i:\n", i, thgrp->group_size);
  kaapi_threadgroup_printinputoutputdata(file, i, &thgrp->ws_khm, &thgrp->ws_vect_input );
  
  for (i=0; i<thgrp->group_size; ++i)
  {
    fprintf(file, "Data on thread %i/%i:\n", i, thgrp->group_size);
    kaapi_threadgroup_printdata(file, i, &thgrp->ws_khm );
  }  
  
  return 0;  
}
