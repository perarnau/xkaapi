/*
** xkaapi
** 
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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
#include <unistd.h>


/** Two orthogonal data structures:
    Numa node -> queue
*/
#define KAAPI_TABLE_QUEUE_NUMA 16
static kaapi_affinity_queue_t* NUMA_queuetable[KAAPI_TABLE_QUEUE_NUMA] = { 
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0
};
static kaapi_cpuset_t* NUMA_cpusettable[KAAPI_TABLE_QUEUE_NUMA];
static int  NUMA_count_queue = 0;


/** Call once to initialize queue at certain level of the hierarchy
    The root always has a queue.
*/
int kaapi_sched_affinity_initialize(void)
{
  /* attach one queue per affinityset of type NUMA Node in the numa hierarchy (even if not used...) */
  for (unsigned short d=0; d < kaapi_default_param.memory.depth; ++d)
  {
    for (unsigned short a=0; a < kaapi_default_param.memory.levels[d].count; ++a)
    {
      kaapi_affinityset_t* level_d = &kaapi_default_param.memory.levels[d].affinity[a];
      if (level_d->type != KAAPI_MEM_NODE) continue;
      if ( (level_d->ncpu > 1) ||(d+1 == kaapi_default_param.memory.depth) )
      {
        kaapi_affinity_queue_t* queue = (kaapi_affinity_queue_t*)malloc( 2*getpagesize() );
        kaapi_sched_initlock(&queue->lock);
        queue->head = 0;
        queue->tail = 0;
        kaapi_allocator_init( &queue->allocator );
        level_d->queue = queue;
        NUMA_queuetable[NUMA_count_queue++] = queue;
        NUMA_cpusettable[level_d->os_index] = &level_d->who;
      }
      else
        level_d->queue = 0;
    }
  } 
  return 0;
}

void kaapi_sched_affinity_destroy(void)
{
}


/*
*/
int kaapi_sched_affinity_binding2mapping(
    kaapi_cpuset_t*              mapping, 
    const kaapi_task_binding_t*  binding,
    const struct kaapi_format_t* task_fmt,
    const struct kaapi_task_t*   task,
    int flag
)
{
  switch (binding->type) {
    case KAAPI_BINDING_ANY:
      kaapi_cpuset_full( mapping );
      return 1;
    
    case KAAPI_BINDING_OCR_ADDR:
    {
      /* lookup for the physical mapping of pages... */
      
      /* FIXEME: should count the number of pages in differents numa node and get the max 
         and not only the numa node of the first page
      */
      kaapi_cpuset_clear(mapping);
      int numanodeid = 0;
      //TODO kaapi_numa_get_page_node( (void*)binding->u.ocr_addr.addr);
      kaapi_assert( (numanodeid >=0) && (numanodeid <KAAPI_TABLE_QUEUE_NUMA) );
      kaapi_cpuset_or( mapping, NUMA_cpusettable[numanodeid] );
      return 1;
    }
      
    case KAAPI_BINDING_OCR_PARAM:
    {
      /* lookup for the physical mapping of pages of some arguments of the task */

      /* FIXEME: should take into account the list of parameters 
         and not only the address of the first one
      */
      void *addr;
      int numanodeid;
      int paramid = 0;
      // TODO kaapi_bitmap_first1_and_zero_64( (kaapi_bitmap_value64_t*)&binding->u.ocr_param.bitmap );
      if (paramid ==0)
      {
        kaapi_cpuset_full( mapping );
        return 0;
      }
      --paramid;
      if ( paramid < (int)kaapi_format_get_count_params( task_fmt, kaapi_task_getargs(task) ) )
      {
        kaapi_cpuset_full( mapping );
        return 0;
      }
      kaapi_cpuset_clear(mapping);
      addr = kaapi_format_get_data_param( task_fmt, paramid, kaapi_task_getargs(task) );
      if (flag !=0) 
      { /* taskdescr task: addr points to the kaapi_data_t */
        addr = __kaapi_pointer2void( ((kaapi_data_t*)addr)->ptr );
      }
      numanodeid = kaapi_numa_get_page_node( addr);
      kaapi_assert( (numanodeid >=0) && (numanodeid <KAAPI_TABLE_QUEUE_NUMA) );
      kaapi_cpuset_or( mapping, NUMA_cpusettable[numanodeid] );
      return 1;
    }
  };
  return 0;
}


/**
*/
kaapi_affinity_queue_t* kaapi_sched_affinity_lookup_queue(kaapi_cpuset_t* mapping)
{
  if (NUMA_count_queue ==0) return 0;

  /* get one value with bit 1 -> the numanode id */
  int numanodeid = kaapi_bitmap_first1_and_zero_128( (kaapi_bitmap_value128_t*)mapping );
  if (numanodeid == 0) return 0;
  --numanodeid; 
  return NUMA_queuetable[numanodeid];
}

/**
*/
kaapi_affinity_queue_t* kaapi_sched_affinity_lookup_numa_queue(int numanodeid)
{
  if (NUMA_count_queue ==0) return 0;
  /* get one value with bit 1 -> the numanode id */
  if (numanodeid == ~0) return 0;
  kaapi_assert_debug( (numanodeid >=0) && (numanodeid <KAAPI_TABLE_QUEUE_NUMA) );
  return NUMA_queuetable[numanodeid];
}

/*
*/
kaapi_affinity_queue_t* kaapi_sched_affinity_random_queue( kaapi_processor_t* kproc )
{
  if (NUMA_count_queue ==0) return 0;
  int r = rand_r( &kproc->seed_data ) % NUMA_count_queue;
  return NUMA_queuetable[r];
}

/*
*/
kaapi_taskdescr_t* kaapi_sched_affinity_allocate_td_dfg( 
    kaapi_affinity_queue_t* queue, 
    kaapi_thread_context_t* thread, 
    kaapi_task_t*           task, 
    const kaapi_format_t*   task_fmt, 
    unsigned int            war_param
)
{
  kaapi_taskdescr_t* td = kaapi_allocator_allocate_td( &queue->allocator, task, task_fmt );
  td->type              = KAAPI_TASKDFG_CASE;
  td->u.dfg.thread      = thread;
  td->u.dfg.war         = war_param;
  return td;
}


/**
  kaapi_affinity_queue_t* queue = _kaapi_get_queue( affinityid );

  kaapi_taskdescr_t* td = kaapi_allocator_allocate_td( &queue->allocator, task, task_fmt);
  td->type   = KAAPI_TASKDFG_CASE;
  td->thread = thread;
  td->war    = war;
*/
int kaapi_sched_affinity_owner_pushtask
(
    kaapi_affinity_queue_t* queue,
    kaapi_taskdescr_t* td
)
{
  if (queue ==0) return ESRCH;
  kaapi_sched_lock( &queue->lock );
  td->next = queue->head;
  td->prev = 0;
  if (queue->head !=0) queue->head->prev = td;
  else queue->head = queue->tail = td;
  kaapi_sched_unlock( &queue->lock );
  return 0;
}


/**
*/
kaapi_taskdescr_t* kaapi_sched_affinity_owner_poptask
(
  kaapi_affinity_queue_t* queue
)
{
  kaapi_taskdescr_t* td;
  if (queue ==0) return 0;

  kaapi_sched_lock( &queue->lock );
  td = queue->head;
  if (td !=0) {
    queue->head = td->next;
    if (queue->head !=0) {
      queue->head->prev = 0;
      td->next = 0;
    }
    else 
      queue->tail = 0;
  }
  kaapi_sched_unlock( &queue->lock );
  return td;
}


/**
*/
int kaapi_sched_affinity_thief_pushtask
(
    kaapi_affinity_queue_t* queue,
    kaapi_taskdescr_t*      td
)
{
  if (queue ==0) return ESRCH;
  kaapi_sched_lock( &queue->lock );
  td->next = 0;
  td->prev = queue->tail;
  if (queue->tail !=0) queue->tail->next = td;
  else queue->head = queue->tail = td;
  kaapi_sched_unlock( &queue->lock );
  return 0;
}


/**
*/
kaapi_taskdescr_t* kaapi_sched_affinity_thief_poptask
(
  kaapi_affinity_queue_t* queue
)
{
  kaapi_taskdescr_t* td;
  if (queue ==0) return 0;
  kaapi_sched_lock( &queue->lock );
  td = queue->tail;
  if (td !=0) {
    queue->tail = td->prev;
    if (queue->tail !=0)
    {
      queue->tail->next = 0;
      td->prev = 0;
    }
    else 
      queue->head = 0;
  }
  kaapi_sched_unlock( &queue->lock );
  return td;
}



/// OLD
#if 0
static void count_range_pages
(unsigned int* counts, uintptr_t addr, size_t size)
{
  if (addr & (0x1000 - 1))
  {
    addr &= ~(0x1000 - 1);
    size += 0x1000;
  }

  if (size & (0x1000 - 1))
    size = (size + 0x1000) & ~(0x1000 - 1);

  for (; size; size -= 0x1000, addr += 0x1000)
    ++counts[(size_t)kaapi_numa_get_addr_binding(addr)];
}

unsigned int kaapi_task_binding_numaid
(const kaapi_task_binding_t* binding)
{
  /* temp, assume valid to do so. */

#if 1

  unsigned int page_counts[8] = {0, };
  size_t i;
  size_t max;

  /* count per node pages */
  for (i = 0; i < binding->u.ocr.count; ++i)
  {
    count_range_pages
      (page_counts, binding->u.ocr.addrs[i], binding->u.ocr.sizes[i]);
  }

  /* find the biggest page count */
  max = 0;
  for (i = 1; i < 8; ++i)
  {
    if (page_counts[max] < page_counts[i])
      max = i;
  }

  return (unsigned int)max;

#else
  /* simple strict mapping */

  return (unsigned int)
    kaapi_numa_get_addr_binding(binding->u.ocr.addrs[0]);

#endif
}


#endif

