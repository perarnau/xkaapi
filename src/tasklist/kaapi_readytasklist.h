/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** joao.lima@imag.fr
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

#ifndef _KAAPI_READYTASKLIST_H_
#define _KAAPI_READYTASKLIST_H_

#define KAAPI_ALLOCATED_TDBLOCSIZE 32 /* such that kaapi_bloctd_t == 1 pagesize */

/** List of pointers, managed by circular buffer.
    First implementation: assume no overflow.
    Assumption: beg <= end   
*/
typedef struct kaapi_onereadytasklist_t {
  long                   size;  /* size of the allocated bloc data or 0 is not allocated */
  kaapi_taskdescr_t**    data;
  kaapi_taskdescr_t*     block[KAAPI_ALLOCATED_TDBLOCSIZE];
  kaapi_workqueue_t      wq;
  kaapi_lock_t           lock;
} kaapi_onereadytasklist_t;


static inline long kaapi_onereadytasklist_getindex(const kaapi_onereadytasklist_t* ortl, long value)
{
  long size = ortl->size;
  if (value <0)
    return (value % size) + size;
  if (value > size)
    return (value % size);
  return value;
}

static inline void kaapi_onereadytasklist_init( kaapi_onereadytasklist_t* ortl )
{
  ortl->data = ortl->block;
  ortl->size = KAAPI_ALLOCATED_TDBLOCSIZE;
  kaapi_atomic_initlock(&ortl->lock);
  kaapi_workqueue_init_with_lock(&ortl->wq, 0, 0, &ortl->lock);
}

static inline void kaapi_onereadytasklist_destroy( kaapi_onereadytasklist_t* ortl )
{
  kaapi_workqueue_destroy(&ortl->wq);
  kaapi_atomic_destroylock(&ortl->lock);
  if (ortl->data != ortl->block) free(ortl->data);
}

static inline int kaapi_onereadytasklist_isempty( const kaapi_onereadytasklist_t* ortl )
{
  return kaapi_workqueue_isempty(&ortl->wq);
}

static inline int kaapi_onereadytasklist_size( const kaapi_onereadytasklist_t* ortl )
{
  return (int)kaapi_workqueue_size(&ortl->wq);
}

static inline int kaapi_onereadytasklist_pop( kaapi_onereadytasklist_t* ortl, kaapi_taskdescr_t** td )
{
  kaapi_workqueue_index_t beg,end;
  
  /* fast check, without lock */
  if( kaapi_onereadytasklist_isempty( ortl ) )
    return EBUSY;

  int retval = kaapi_workqueue_pop(&ortl->wq, &beg, &end, 1);
  if (retval == 0)
  {
    long index = kaapi_onereadytasklist_getindex(ortl, beg);
    *td = ortl->data[ index ];
    ortl->data[ index ] = 0;
    if (*td ==0) retval = EBUSY;
  }
  return retval;  
}


static inline int kaapi_onereadytasklist_realloc( kaapi_onereadytasklist_t* ortl )
{
  /* realloc */
  size_t newsize = 2*ortl->size;
  kaapi_taskdescr_t** olddata = ortl->data;
  kaapi_taskdescr_t** newdata = (kaapi_taskdescr_t**)malloc(newsize*sizeof(kaapi_taskdescr_t*));
  memcpy(newdata, ortl->data, ortl->size*sizeof(kaapi_taskdescr_t*));
  /* change original */
  kaapi_atomic_lock(&ortl->lock);
  ortl->data = newdata;
  ortl->size = newsize;
  kaapi_atomic_unlock(&ortl->lock);
  if (olddata != ortl->block) free(olddata);
  return 0;
}

static inline int kaapi_onereadytasklist_push( kaapi_onereadytasklist_t* ortl, kaapi_taskdescr_t* td )
{
  kaapi_workqueue_index_t beg = kaapi_workqueue_range_begin(&ortl->wq) -1;
  kaapi_workqueue_index_t end = kaapi_workqueue_range_end(&ortl->wq);
  
  if (end-beg >= ortl->size)
    kaapi_onereadytasklist_realloc(ortl);
  
  ortl->data[ kaapi_onereadytasklist_getindex(ortl,beg) ] = td;
  kaapi_workqueue_push(&ortl->wq, beg );
  return 0;
}

static inline int kaapi_onereadytasklist_remote_push( kaapi_onereadytasklist_t* ortl, kaapi_taskdescr_t* td )
{
  kaapi_atomic_lock( &ortl->lock );
  kaapi_workqueue_index_t end = kaapi_workqueue_range_end(&ortl->wq);
  kaapi_workqueue_index_t beg = kaapi_workqueue_range_begin(&ortl->wq);

  if (end-beg >= ortl->size)
    kaapi_onereadytasklist_realloc(ortl);

  ortl->data[ kaapi_onereadytasklist_getindex(ortl,end) ] = td;
  kaapi_workqueue_rpush( &ortl->wq, end+1 );
  kaapi_atomic_unlock( &ortl->lock );
  return 0;
}

static inline int kaapi_onereadytasklist_steal( kaapi_onereadytasklist_t* ortl, kaapi_taskdescr_t** td ) 
{
  int size_ws;
  kaapi_workqueue_index_t beg,end;
  int retval;
  
  size_ws = kaapi_onereadytasklist_size( ortl );
  if (size_ws ==0) 
    return ERANGE;

  /* */
  kaapi_atomic_lock( &ortl->lock );
  retval = kaapi_workqueue_steal(&ortl->wq, &beg, &end, 1);
  if (retval ==0)
  {
    long index = kaapi_onereadytasklist_getindex(ortl, beg);
    *td = ortl->data[ index ];
    ortl->data[ index ] = 0;
    if (*td ==0) retval = EBUSY;
  }
  kaapi_atomic_unlock( &ortl->lock );
  return retval;
}

/** The workqueue of ready tasks with priorities
 */
typedef struct kaapi_readytasklist_t {
  kaapi_onereadytasklist_t prl[KAAPI_TASKLIST_NUM_PRIORITY]; 
  kaapi_atomic_t	    cnt_tasks;
} kaapi_readytasklist_t;

/* 
 * max_prio - maximum priority of a given kproc
 * min_prio - minimum priority of a given kproc
 * inc_prio - increment 
 */
static inline void
kaapi_readylist_get_priority_range(
                                   int* const min_prio, int* const max_prio, int* const inc_prio )
{
  if( kaapi_processor_get_type( kaapi_get_current_processor() ) == KAAPI_PROC_TYPE_CUDA )
  {
    *min_prio = (KAAPI_TASKLIST_GPU_MIN_PRIORITY+1);
    *max_prio = KAAPI_TASKLIST_GPU_MAX_PRIORITY;
    *inc_prio = 1;
  } else {
    *min_prio = (KAAPI_TASKLIST_CPU_MIN_PRIORITY-1);
    *max_prio = KAAPI_TASKLIST_CPU_MAX_PRIORITY;
    *inc_prio = -1;
  }
}

/*
 */
static inline int kaapi_readytasklist_init( 
                                           kaapi_readytasklist_t* rtl
/* unused                                           kaapi_lock_t*          lock */
)
{
  int i;
  
  for (i =0; i<KAAPI_TASKLIST_NUM_PRIORITY; ++i) 
    kaapi_onereadytasklist_init( &rtl->prl[i] );
  
  KAAPI_ATOMIC_WRITE( &rtl->cnt_tasks, 0 );
  return 0;
}

static inline int kaapi_readytasklist_destroy( kaapi_readytasklist_t* rtl )
{
  int i;
  
  for (i =0; i<KAAPI_TASKLIST_NUM_PRIORITY; ++i) 
    kaapi_onereadytasklist_destroy( &rtl->prl[i] );
  return 0;
}

static inline int kaapi_readytasklist_isempty( kaapi_readytasklist_t* rtl )
{
  return (KAAPI_ATOMIC_READ( &rtl->cnt_tasks ) == 0 );
}

static inline int kaapi_readylist_push( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t* td, int priority )
{
  kaapi_onereadytasklist_t* ortl;
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MAX_PRIORITY) && (priority <= KAAPI_TASKLIST_MIN_PRIORITY) );
  
  ortl = &rtl->prl[priority];
  kaapi_onereadytasklist_push( ortl, td );
  KAAPI_ATOMIC_INCR( &rtl->cnt_tasks );
  return priority;
}

static inline int kaapi_readylist_remote_push( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t* td, int priority )
{
  kaapi_onereadytasklist_t* ortl;
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MAX_PRIORITY) && (priority <= KAAPI_TASKLIST_MIN_PRIORITY) );
  
  ortl = &rtl->prl[priority];
  kaapi_onereadytasklist_remote_push( ortl, td );
  KAAPI_ATOMIC_INCR( &rtl->cnt_tasks );
  return priority;
}

/** Return the number of tasks inside the tasklist
 */
static inline size_t kaapi_readylist_workload( kaapi_readytasklist_t* rtl )
{
  return KAAPI_ATOMIC_READ( &rtl->cnt_tasks );
}

static inline int kaapi_readylist_steal( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** td ) 
{
  kaapi_onereadytasklist_t* onertl;
  int err;
  int prio;
  int max_prio= 0, min_prio= 0, inc_prio= 0;
  
  if( KAAPI_ATOMIC_READ( &rtl->cnt_tasks) == 0 )
    return 1;
  
  kaapi_readylist_get_priority_range( &min_prio, &max_prio, &inc_prio );
  for( prio = max_prio; prio != min_prio; prio += inc_prio ) 
  {
    onertl = &rtl->prl[prio];
    err = kaapi_onereadytasklist_steal( onertl, td );
    if( err == 0 )
    {
	    KAAPI_ATOMIC_DECR( &rtl->cnt_tasks );
	    return 0;
    }
  }
  return 1;
}

static inline int kaapi_readylist_pop( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** td )
{
  kaapi_onereadytasklist_t* onertl;
  int prio;
  int err;
  int max_prio= 0, min_prio= 0, inc_prio= 0;
  
  if( KAAPI_ATOMIC_READ( &rtl->cnt_tasks) == 0 )
    return 1;
  
  kaapi_readylist_get_priority_range( &min_prio, &max_prio, &inc_prio );
  for( prio = max_prio; prio != min_prio; prio += inc_prio ) 
  {
    onertl = &rtl->prl[prio];
    err = kaapi_onereadytasklist_pop( onertl, td );
    if( err == 0 ){
	    KAAPI_ATOMIC_DECR( &rtl->cnt_tasks );
	    return 0;
    } else
	    if( err != EBUSY )
        return err;
  }
  return EBUSY;
}


#endif /* _KAAPI_READYTASKLIST_H_ */ 
