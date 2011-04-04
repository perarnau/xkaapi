/*
** kaapi_thread_print.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
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
#include <stdio.h>
#include <inttypes.h>


static int noprint_activationlink = 0;
static int noprint_versionlink = 0;

/**/
static inline void _kaapi_print_data( FILE* file, const void* ptr, unsigned long version)
{
  fprintf(file,"%lu00%lu [label=\"%p v%lu\", shape=box, style=filled, color=steelblue];\n", 
      version, (uintptr_t)ptr, ptr,version-1 
  );
}

/**/
static inline void _kaapi_print_activation_link(
    FILE* file,
    const kaapi_taskdescr_t* td_src,
    const kaapi_taskdescr_t* td_dest
)
{
  if (!noprint_activationlink)
    fprintf(file,"%lu -> %lu [arrowhead=halfopen, style=filled, color=red];\n", 
      (uintptr_t)td_src->task, (uintptr_t)td_dest->task );
}


/**/
static inline void _kaapi_print_write_edge( 
    FILE* file, 
    kaapi_task_t* task, 
    const void* ptr, 
    unsigned long version, 
    kaapi_access_mode_t m
)
{
  if (KAAPI_ACCESS_IS_READWRITE(m))
    fprintf(file,"%lu -> %lu00%lu[dir=both,arrowtail=diamond,arrowhead=vee];\n", 
        (uintptr_t)task, version, (uintptr_t)ptr );
  else if (KAAPI_ACCESS_IS_CUMULWRITE(m))
  { /* the final version will have tag version+1, not yet generated */
    fprintf(file,"%lu -> %lu00%lu[dir=both,arrowtail=inv,arrowhead=tee];\n", 
        (uintptr_t)task, version+1, (uintptr_t)ptr ); 
    return;
  } 
  else 
    fprintf(file,"%lu -> %lu00%lu;\n", (uintptr_t)task, version, (uintptr_t)ptr );

  if (!noprint_versionlink)
  {
    /* add version edge */
    if (version >1)
      fprintf(file,"%lu00%lu -> %lu00%lu [style=dotted];\n", 
          version-1, (uintptr_t)ptr, version, (uintptr_t)ptr );
  }

}

/**/
static inline void _kaapi_print_read_edge( 
    FILE* file, 
    kaapi_task_t* task, 
    const void* ptr, 
    unsigned long version, 
    kaapi_access_mode_t m
)
{
  if (KAAPI_ACCESS_IS_READWRITE(m))
    fprintf(file,"%lu00%lu -> %lu[dir=both,arrowtail=diamond,arrowhead=vee];\n", version, (uintptr_t)ptr, (uintptr_t)task );
  else 
    fprintf(file,"%lu00%lu -> %lu;\n", version, (uintptr_t)ptr, (uintptr_t)task );
}

/**/
static inline void _kaapi_print_tag_node( 
    FILE* file, 
    unsigned long tag
)
{
  fprintf(file,"tag_%lu [shape=Mcircle, label=\"%lu\"];\n", tag, tag );
}


/**/
static inline void _kaapi_print_task( 
    FILE* file, 
    kaapi_hashmap_t* data_khm,
    kaapi_task_t* task, 
    const kaapi_taskdescr_t* td,
    const char* sec_name
)
{
  kaapi_activationlink_t* lk;
  kaapi_taskdescr_t* tda;
  char bname[128];
  const char* fname = "<empty format>";
  const char* shape = "ellipse";
  const kaapi_format_t* fmt;
  kaapi_task_body_t body;

  void* sp = task->sp;
  body = kaapi_task_getbody(task);
  if ((td !=0) && (td->fmt !=0))
  {
    fmt = td->fmt;
    fname = fmt->name;
  } 
  else 
  {
    fmt = kaapi_format_resolvebybody( body );
    if (fmt !=0)
      fname = fmt->name;
  }
  /* specialize shape / name for some well knowns tasks */
  if (body == kaapi_taskstartup_body)
  {
      fname = "startup";
      shape = "tripleoctagon";
  } 
  else if (body == kaapi_taskmain_body)
  {
      fname = "maintask";
      shape = "doubleoctagon";
  } 
  else if ((body == kaapi_taskmove_body) || (body == kaapi_taskalloc_body) || (body == kaapi_taskfinalizer_body))
  {
    if (body == kaapi_taskfinalizer_body)
    {
      fname = "finalizer";
      shape = "invtrapezium";
    }
    else {
      shape = "diamond";
      if (body == kaapi_taskmove_body)
        fname = "move";
      else if (body == kaapi_taskalloc_body)
        fname = "alloc";
    }
    kaapi_move_arg_t* argtask = (kaapi_move_arg_t*)sp;
    kaapi_hashentries_t* entry = kaapi_hashmap_findinsert(data_khm, argtask->dest);
    if (entry->u.data.tag ==0)
    {
      /* display the node */
      entry->u.data.tag = 1;
      _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
    }
    if (body != kaapi_taskfinalizer_body)
    {
      _kaapi_print_write_edge( file, td->task, entry->key, entry->u.data.tag, KAAPI_ACCESS_MODE_W );
    }
    else {
      _kaapi_print_data( file, entry->key, (int)entry->u.data.tag+1 );
      if (!noprint_versionlink)
      {
        /* add version edge */
        fprintf(file,"%lu00%lu -> %lu00%lu [style=dotted];\n", 
              (unsigned long)entry->u.data.tag, (uintptr_t)entry->key, (unsigned long)entry->u.data.tag+1, (uintptr_t)entry->key );
      }
      _kaapi_print_read_edge( file, td->task, entry->key, entry->u.data.tag+1, KAAPI_ACCESS_MODE_R );
      entry->u.data.tag += 2;
      _kaapi_print_write_edge( file, td->task, entry->key, entry->u.data.tag, KAAPI_ACCESS_MODE_W );
      _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
    }
    
  }
  else if (body == kaapi_taskbcast_body)
  {
    shape = "octagon";
    kaapi_bcast_arg_t* argtask = (kaapi_bcast_arg_t*)sp;
    snprintf(bname, 128, "bcast\\ntag:%lu", (unsigned long)argtask->tag );
    fname = bname;
    kaapi_hashentries_t* entry = kaapi_hashmap_findinsert(data_khm, argtask->src);
    if (entry->u.data.tag ==0)
    {
      /* display the node */
      entry->u.data.tag = 1;
      _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
    }
    _kaapi_print_read_edge( file, td->task, entry->key, entry->u.data.tag, KAAPI_ACCESS_MODE_R );
//    fprintf(file,"%lu -> tag_%lu  [style=bold, color=blue, label=\"tag:%lu\"];\n", 
//      (uintptr_t)task, (uintptr_t)argtask->tag, (unsigned long)argtask->tag );
   if (sec_name !=0)
     fprintf(file,"}\n");
  _kaapi_print_tag_node(file, (unsigned long)argtask->tag);
  fprintf(file,"%lu -> tag_%lu  [style=bold, color=blue];\n", 
    (uintptr_t)task, (uintptr_t)argtask->tag );
   if (sec_name !=0)
     fprintf(file, "%s", sec_name);
  }
  else if (body == kaapi_taskrecv_body)
  {
    shape = "invtrapezium";
    kaapi_recv_arg_t* argtask = (kaapi_recv_arg_t*)sp;
    snprintf(bname, 128, "recv\\ntag:%lu", (unsigned long)argtask->tag );
    fname = bname;
    kaapi_hashentries_t* entry = kaapi_hashmap_findinsert(data_khm, argtask->dest);
    if (entry->u.data.tag ==0)
    {
      /* display the node */
      entry->u.data.tag = 1;
      _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
    }
    _kaapi_print_write_edge( file, td->task, entry->key, entry->u.data.tag, KAAPI_ACCESS_MODE_W );
//    fprintf(file,"tag_%lu -> %lu [style=bold, color=blue, label=\"tag:%lu\"];\n", 
//      (uintptr_t)argtask->tag, (uintptr_t)task, (unsigned long)argtask->tag );
   if (sec_name !=0)
     fprintf(file,"}\n");
    _kaapi_print_tag_node(file, argtask->tag);
    fprintf(file,"tag_%lu -> %lu [style=bold, color=blue];\n", 
      (uintptr_t)argtask->tag, (uintptr_t)task );
   if (sec_name !=0)
     fprintf(file, "%s", sec_name);
  }

  if (td ==0) 
  {
    fprintf( file, "%lu [label=\"%s\\n task=%p\", shape=%s, style=filled, color=orange];\n", 
      (uintptr_t)task, fname, (void*)task, shape
    );
    return;
  }

  fprintf( file, "%lu [label=\"%s\\n task=%p\\n depth=%" PRIu64
           "\\n wc=%i, counter=%i, \", shape=%s, style=filled, color=orange];\n", 
    (uintptr_t)task, fname, (void*)task, 
    td->date, 
    (int)td->wc,
    (int)KAAPI_ATOMIC_READ(&td->counter),
    shape
  );

  /* display activation link */
  lk = td->list.front;
  while (lk !=0)
  {
    tda = lk->td;
    _kaapi_print_activation_link( file, td, tda );
    lk = lk->next;
  }  

  /* display bcast link */
  if (td->bcast !=0)
  {
    lk = td->bcast->front;
    while (lk !=0)
    {
      tda = lk->td;
      _kaapi_print_activation_link( file, td, tda );
      lk = lk->next;
    }
  }

  if (fmt ==0)
    return;
  
  size_t count_params = kaapi_format_get_count_params(fmt, sp );

  for (unsigned int i=0; i < count_params; i++) 
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(fmt, i, sp) );
    if (m == KAAPI_ACCESS_MODE_V) 
      continue;
    
    /* its an access */
    kaapi_access_t access = kaapi_format_get_access_param(fmt, i, sp);

    /* find the version info of the data using the hash map */
    kaapi_hashentries_t* entry = kaapi_hashmap_findinsert(data_khm, access.data);
    if (entry->u.data.tag ==0)
    {
      /* display the node */
      entry->u.data.tag = 1;
      _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
    }
    
    /* display arrow */
    if (KAAPI_ACCESS_IS_READ(m))
    {
      _kaapi_print_read_edge( file, td->task, entry->key, entry->u.data.tag, m  );
    }
    if (KAAPI_ACCESS_IS_WRITE(m))
    {
      entry->u.data.tag++;
      /* display new version */
      _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
      _kaapi_print_write_edge( file, td->task, entry->key, entry->u.data.tag, m );
    }
    if (KAAPI_ACCESS_IS_CUMULWRITE(m))
    {
      _kaapi_print_write_edge( file, td->task, entry->key, entry->u.data.tag, m );
    }
  }
  
}


/**
*/
typedef struct {
  kaapi_hashmap_t data_khm;
  FILE*           file;
  const char*     sec_name;
} _kaapi_print_context;


/**
*/
static void _kaapi_print_task_executor( kaapi_taskdescr_t* td, void* arg )
{
  _kaapi_print_context* ctxt = (_kaapi_print_context*)arg;
  _kaapi_print_task( ctxt->file, &ctxt->data_khm, td->task, td, ctxt->sec_name );
}


/** 
*/
int kaapi_thread_tasklist_print_dot  ( FILE* file, const kaapi_tasklist_t* tasklist, int clusterflags )
{
  _kaapi_print_context the_hash_map;
  char sec_name[128];
  
  if (tasklist ==0) return 0;

  noprint_activationlink = (0 != getenv("KAAPI_DOT_NOACTIVATION_LINK"));
  noprint_versionlink = (0 != getenv("KAAPI_DOT_NOVERSION_LINK"));

  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &the_hash_map.data_khm, 0 );
  the_hash_map.file = file;
  the_hash_map.sec_name =0;

  if (clusterflags !=0)
  {
    sprintf(sec_name, "subgraph cluster_%lu {\n",(uintptr_t)tasklist);
    the_hash_map.sec_name = sec_name;
    fprintf(file, "%s\n", sec_name);
  }
  else
    fprintf(file, "digraph G {\n");


  kaapi_thread_abstractexec_readylist( tasklist, _kaapi_print_task_executor, &the_hash_map );

  fprintf(file, "\n\n}\n");
  fflush(file);

  kaapi_hashmap_destroy(&the_hash_map.data_khm);
  return 0;
}


/** Print the frame without taking into account the tasklist
*/
static int kaapi_frame_print_dot_notasklist  ( FILE* file, const kaapi_frame_t* frame )
{
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  kaapi_hashmap_t data_khm;
  
  if (frame ==0) return 0;

  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &data_khm, 0 );
  
  fprintf(file, "digraph G {\n");
  
  task_top = frame->pc;
  task_bot = frame->sp;
  while (task_top > task_bot) 
  {
    _kaapi_print_task( file, &data_khm, task_top, 0, 0 );
    --task_top;
  }
  fprintf(file, "\n}\n");
  fflush(file);

  kaapi_hashmap_destroy(&data_khm);
  return 0;
}


/** 
*/
int kaapi_frame_print_dot  ( FILE* file, const kaapi_frame_t* frame, int flag )
{
  if ((file ==0) || (frame ==0)) return EINVAL;
  if (frame->tasklist !=0) 
    return kaapi_thread_tasklist_print_dot(file, frame->tasklist, flag);
  return kaapi_frame_print_dot_notasklist(file, frame);
}
