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


/**/
static inline void _kaapi_print_data( FILE* file, const void* ptr, unsigned long version)
{
  fprintf(file,"%lu00%lu [label=\"%p v%lu\", shape=box, style=filled, color=steelblue];\n", 
      version, (uintptr_t)ptr, ptr,version-1 
  );
}

/**/
static inline void _kaapi_print_task( 
    FILE* file, 
    kaapi_hashmap_t* khm,
    kaapi_hashmap_t* khm_data,
    kaapi_task_t* task, 
    const char* fname, 
    const kaapi_taskdescr_t* td, 
    const char* shape
)
{
  kaapi_activationlink_t* lk;
  kaapi_taskdescr_t* tda;

  if (td ==0) 
  {
    fprintf( file, "%lu \[label=\"%s\\n task=%p\", shape=%s, style=filled, color=orange];\n", 
      (uintptr_t)task, fname, (void*)task, shape
    );
    return;
  }

  fprintf( file, "%lu \[label=\"%s\\n task=%p\\n depth=%llu\\n wc=%i\", shape=%s, style=filled, color=orange];\n", 
    (uintptr_t)task, fname, (void*)task, 
    td->date, 
    (int)KAAPI_ATOMIC_READ(&td->counter),
    shape
  );
  
  /* display activation link */
  lk = td->list.front;
  while (lk !=0)
  {
    tda = lk->td;
    fprintf(file,"%lu -> %lu [arrowhead=halfopen, style=filled, color=red];\n", (uintptr_t)task, (uintptr_t)tda->task );
    /* print the finalizer only once */
    if (kaapi_task_getbody(tda->task) == kaapi_taskfinalizer_body)
    {
      kaapi_hashentries_t* entry;
      kaapi_taskdescr_t* td_finalizer;
      entry = kaapi_hashmap_find( khm, tda->task );
      kaapi_assert(entry !=0);
      td_finalizer = (kaapi_taskdescr_t*)entry->u.data.ptr;
      if (entry->u.data.tag ==1)
      {
        entry->u.data.tag = 2;
        fprintf( file, "%lu \[label=\"%s\\n task=%p\\n depth=%llu\\n wc=%i\", shape=%s, style=filled, color=orange];\n", 
          (uintptr_t)td_finalizer->task, "finalizer", (void*)td_finalizer->task, 
          td_finalizer->date, 
          (int)KAAPI_ATOMIC_READ(&td_finalizer->counter),
          shape
        );
        
        /* display also an array to the data */
        kaapi_move_arg_t* arg = (kaapi_move_arg_t*)td_finalizer->task->sp;
        entry = kaapi_hashmap_find( khm_data, arg->dest );
        kaapi_assert( entry !=0 );
        _kaapi_print_data(file, entry->key, entry->u.data.tag+1);
        fprintf(file,"%lu -> %lu00%lu;\n", 
          (uintptr_t)td_finalizer->task, entry->u.data.tag+1, (uintptr_t)entry->key );
        fprintf(file,"%lu00%lu -> %lu;\n", entry->u.data.tag, (uintptr_t)entry->key,
           (uintptr_t)td_finalizer->task 
        );
        fprintf(file,"%lu00%lu -> %lu00%lu [style=dotted];\n", 
            entry->u.data.tag, (uintptr_t)entry->key, entry->u.data.tag+1, (uintptr_t)entry->key );
      }
    }
    lk = lk->next;
  }  
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

  /* add version edge */
  if (version >1)
    fprintf(file,"%lu00%lu -> %lu00%lu [style=dotted];\n", 
        version-1, (uintptr_t)ptr, version, (uintptr_t)ptr );
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



/** Visit tasklist data structure to keep correspondance between
    kaapi_task_t and kaapi_taskdesc_t
*/
/**
*/
static int _kaapi_add_unvisited_td( kaapi_hashmap_t* khm, kaapi_activationlink_t* lk)
{
  kaapi_hashentries_t* entry;

  if (lk ==0) return 0;
  kaapi_taskdescr_t* tda;
  while (lk !=0)
  {
    tda = lk->td;
    entry = kaapi_hashmap_findinsert(khm, tda->task);
    if (entry->u.data.tag ==0)
    {
      entry->u.data.tag = 1; /* task inserted */
      entry->u.data.ptr = tda;
      if (tda->list.front !=0)
        _kaapi_add_unvisited_td(khm, tda->list.front);
    }
    lk = lk->next;
  }
  return 0;
}

/**
*/
static int _kaapi_add_td( kaapi_hashmap_t* khm, kaapi_tasklist_t* tl )
{
  /* new history of visited data */
  kaapi_hashentries_t* entry;

  kaapi_taskdescr_t* td = tl->front;
  while (td != 0)
  {
    entry = kaapi_hashmap_findinsert(khm, td->task);
    if (entry->u.data.tag ==0)
    { /* first time I visit it: print and insert activated task descr into the hashmap */
      entry->u.data.tag = 1; /* task print */
      entry->u.data.ptr = td; 

      /* add other td */
      _kaapi_add_unvisited_td( khm, td->list.front );
    }
    td = td->next;
  }
  return 0;  
}


/** 
*/
static int kaapi_frame_print_dot_tasklist  ( FILE* file, kaapi_frame_t* frame )
{
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  kaapi_task_body_t body;
  const kaapi_format_t* fmt;
  const char* fname;
  kaapi_taskdescr_t* td = 0;
  kaapi_hashentries_t* entry;
  kaapi_hashmap_t visit_khm;
  kaapi_hashmap_t td_khm;             /* store the ready task descriptor */
  
  if (frame ==0) return 0;

  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &visit_khm, 0 );
  
  kaapi_assert(frame->tasklist !=0);
  kaapi_hashmap_init( &td_khm, 0 );
  _kaapi_add_td( &td_khm, frame->tasklist );

  fprintf(file, "digraph G {\n");
  
  /* search if task move or alloc are in the stack 
     Print them before because they are at the begining of each chain shared data
  */
  fname = "";
  task_top = frame->pc;
  task_bot = frame->sp;
  while (task_top > task_bot)
  {
    body = kaapi_task_getbody(task_top);
    if ((body == kaapi_taskmove_body) || (body == kaapi_taskalloc_body))
    {
      if (body == kaapi_taskmove_body)
        fname = "move";
      else if (body == kaapi_taskalloc_body)
        fname = "alloc";

      td = 0;
      entry = kaapi_hashmap_find( &td_khm, task_top );
      kaapi_assert(entry !=0);
      td = (kaapi_taskdescr_t*)entry->u.data.ptr;
      _kaapi_print_task( file, 0, 0, task_top, fname, td, "diamond" );

      kaapi_move_arg_t* arg= (kaapi_move_arg_t*)task_top->sp;
      entry = kaapi_hashmap_findinsert(&visit_khm, arg->dest);
      if (entry->u.data.tag ==0)
      {
        /* display the node */
        entry->u.data.tag = 1;
        entry->u.data.last_mode = KAAPI_ACCESS_MODE_W;
        _kaapi_print_data( file, entry->key, entry->u.data.tag  );
      }
      /* display the edge */
      _kaapi_print_write_edge( file, task_top, entry->key, entry->u.data.tag, KAAPI_ACCESS_MODE_W  );
    }
    --task_top;
  }
  
  
  /* print the rest of the dfg */
  task_top = frame->pc;
  task_bot = frame->sp;
  while (task_top > task_bot) 
  {
    body = kaapi_task_getbody(task_top);
    
    /* do not print as normal task these tasks because they are pushed at the end of the stack
       and we cannot use the creation order (order in the stack) to detect dependencies
    */
    if ((kaapi_task_getbody(task_top) == kaapi_taskmove_body) ||
        (kaapi_task_getbody(task_top) == kaapi_taskalloc_body) ||
        (kaapi_task_getbody(task_top) == kaapi_taskfinalizer_body) )
    {
      --task_top;
      continue;
    }
    
    fmt = kaapi_format_resolvebybody( body );
    
    if (fmt ==0) 
    {
      fname = "<empty format>";
      if (body == kaapi_nop_body) 
        fname = "nop";
      else if (body == kaapi_taskstartup_body) 
      {
        fname = "startup";
        _kaapi_print_task( file, 0, 0, task_top, fname, 0, "tripleoctagon" );
      }
      else if ( body == kaapi_taskmain_body) 
      {
        fname = "maintask";
        _kaapi_print_task( file, 0, 0, task_top, fname, 0, "doubleoctagon" );
      }
      --task_top;
      continue;
    }
    else {
        fname = fmt->name;
    }

    entry = kaapi_hashmap_find( &td_khm, task_top );
    kaapi_assert(entry !=0);
    td = (kaapi_taskdescr_t*)entry->u.data.ptr;

    /* create the node with the task */
    _kaapi_print_task( file, &td_khm, &visit_khm, task_top, fname, td, "ellipse" );

    
    /* display dependency of the task with its shared arguments */
    void* sp = task_top->sp;
    size_t count_params = kaapi_format_get_count_params(fmt, sp );

    for (unsigned int i=0; i < count_params; i++) 
    {
      kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(fmt, i, sp) );
      if (m == KAAPI_ACCESS_MODE_V) 
        continue;
      
      /* its an access */
      kaapi_access_t access = kaapi_format_get_access_param(fmt, i, sp);

      /* find the version info of the data using the hash map */
      entry = kaapi_hashmap_findinsert(&visit_khm, access.data);
      if (entry->u.data.tag ==0)
      {
        /* display the node */
        entry->u.data.tag = 1;
        entry->u.data.last_mode = m;
        _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
      }
      
      /* display arrow */
      if (KAAPI_ACCESS_IS_READ(m))
      {
        if (KAAPI_ACCESS_IS_CUMULWRITE(entry->u.data.last_mode))
        {
          entry->u.data.tag++;
          _kaapi_print_data( file, entry->key, (int)entry->u.data.tag);
        }
        _kaapi_print_read_edge( file, task_top, entry->key, entry->u.data.tag, m  );
      }
      if (KAAPI_ACCESS_IS_WRITE(m))
      {
        if (KAAPI_ACCESS_IS_CUMULWRITE(entry->u.data.last_mode))
        {
          entry->u.data.tag++;
          _kaapi_print_data( file, entry->key, (int)entry->u.data.tag);
        }
        entry->u.data.tag++;
        /* display new version */
        _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
        _kaapi_print_write_edge( file, task_top, entry->key, entry->u.data.tag, m );
      }
      else if (KAAPI_ACCESS_IS_CUMULWRITE(m))
      {
        /* write a cumul edge to the next version */
        _kaapi_print_write_edge( file, task_top, entry->key, entry->u.data.tag, m );        
      }
      entry->u.data.last_mode = m;
    }

    --task_top;
  }
  fprintf(file, "\n}\n");
  fflush(file);

  kaapi_hashmap_destroy(&visit_khm);
  kaapi_hashmap_destroy(&td_khm);
  return 0;
}


/** Print the frame without taking into account the tasklist
*/
static int kaapi_frame_print_dot_notasklist  ( FILE* file, kaapi_frame_t* frame )
{
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  kaapi_task_body_t body;
  const kaapi_format_t* fmt;
  const char* fname;
  kaapi_hashentries_t* entry;
  kaapi_hashmap_t visit_khm;
  
  if (frame ==0) return 0;

  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &visit_khm, 0 );
  
  fprintf(file, "digraph G {\n");
  
  /* search if task move are in the stack */
  fname = "";
  task_top = frame->pc;
  task_bot = frame->sp;
  while (task_top > task_bot)
  {
    body = kaapi_task_getbody(task_top);
    if ((body == kaapi_taskmove_body) || (body == kaapi_taskalloc_body))
    {
      if (body == kaapi_taskmove_body)
        fname = "move";
      else if (body == kaapi_taskalloc_body)
        fname = "alloc";

      _kaapi_print_task( file, 0, 0, task_top, fname, 0, "diamond" );

      kaapi_move_arg_t* arg= (kaapi_move_arg_t*)task_top->sp;
      entry = kaapi_hashmap_findinsert(&visit_khm, arg->dest);
      if (entry->u.data.tag ==0)
      {
        /* display the node */
        entry->u.data.tag = 1;
        _kaapi_print_data( file, entry->key, entry->u.data.tag  );
      }
      /* display the edge */
      _kaapi_print_write_edge( file, task_top, entry->key, entry->u.data.tag, KAAPI_ACCESS_MODE_W  );
    }
    --task_top;
  }
  
  
  task_top = frame->pc;
  task_bot = frame->sp;
  while (task_top > task_bot) 
  {
    body = kaapi_task_getbody(task_top);
    
    if ((kaapi_task_getbody(task_top) != kaapi_taskmove_body) &&
        (kaapi_task_getbody(task_top) != kaapi_taskalloc_body) )
    {
      --task_top;
      continue;
    }
    
    fmt = kaapi_format_resolvebybody( body );
    
    if (fmt ==0) 
    {
      fname = "<empty format>";
      if (body == kaapi_nop_body) 
        fname = "nop";
      else if (body == kaapi_taskstartup_body) 
      {
        fname = "startup";
        _kaapi_print_task( file, 0, 0, task_top, fname, 0, "tripleoctagon" );
      }
      else if ( body == kaapi_taskmain_body) 
      {
        fname = "maintask";
        _kaapi_print_task( file, 0, 0, task_top, fname, 0, "doubleoctagon" );
      }
      else if ( body == kaapi_taskfinalizer_body) 
      {
        fname = "finalizer";
        _kaapi_print_task( file, 0, 0, task_top, fname, 0, "invtrapezium" );
      }
      --task_top;
      continue;
    }
    else {
        fname = fmt->name;
    }
    /* create the node with the task */
    _kaapi_print_task( file, 0, 0, task_top, fname, 0, "ellipse" );

    /* display dependency of the task with its shared arguments */
    void* sp = task_top->sp;
    size_t count_params = kaapi_format_get_count_params(fmt, sp );

    for (unsigned int i=0; i < count_params; i++) 
    {
      kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(fmt, i, sp) );
      if (m == KAAPI_ACCESS_MODE_V) 
        continue;
      
      /* its an access */
      kaapi_access_t access = kaapi_format_get_access_param(fmt, i, sp);

      /* find the version info of the data using the hash map */
      entry = kaapi_hashmap_findinsert(&visit_khm, access.data);
      if (entry->u.data.tag ==0)
      {
        /* display the node */
        entry->u.data.tag = 1;
        _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
      }
      
      /* display arrow */
      if (KAAPI_ACCESS_IS_READ(m))
      {
        _kaapi_print_read_edge( file, task_top, entry->key, entry->u.data.tag, m  );
      }
      if (KAAPI_ACCESS_IS_WRITE(m))
      {
        entry->u.data.tag++;
        /* display new version */
        _kaapi_print_data( file, entry->key, (int)entry->u.data.tag );
        _kaapi_print_write_edge( file, task_top, entry->key, entry->u.data.tag, m );
      }
      if (KAAPI_ACCESS_IS_CUMULWRITE(m))
      {
        _kaapi_print_write_edge( file, task_top, entry->key, entry->u.data.tag, m );
      }
    }

    --task_top;
  }
  fprintf(file, "\n}\n");
  fflush(file);

  kaapi_hashmap_destroy(&visit_khm);
  return 0;
}


/** 
*/
int kaapi_frame_print_dot  ( FILE* file, kaapi_frame_t* frame )
{
  if ((file ==0) || (frame ==0)) return EINVAL;
  if (frame->tasklist !=0) 
    return kaapi_frame_print_dot_tasklist(file, frame);
  return kaapi_frame_print_dot_notasklist(file, frame);
}
