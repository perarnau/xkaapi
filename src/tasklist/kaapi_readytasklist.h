
#ifndef _KAAPI_READYTASKLIST_H_
#define _KAAPI_READYTASKLIST_H_

/** One workqueue of ready tasks 
*/
typedef struct kaapi_onereadytasklist_t {
    kaapi_lock_t	    lock;
    kaapi_taskdescr_t*	    head;
    kaapi_taskdescr_t*	    tail;
    int		size;
} kaapi_onereadytasklist_t;

static inline void kaapi_onereadytasklist_init( kaapi_onereadytasklist_t* ortl )
{
    ortl->size = 0;
    ortl->head = ortl->tail = NULL;
    kaapi_atomic_initlock( &ortl->lock );
}

static inline void kaapi_onereadytasklist_destroy( kaapi_onereadytasklist_t* ortl )
{
    ortl->size = 0;
    ortl->head = ortl->tail = NULL;
    kaapi_atomic_destroylock( &ortl->lock ); 
}

/**/
static inline int kaapi_onereadytasklist_isempty( const kaapi_onereadytasklist_t* ortl )
{
    return ( ortl->size == 0 );
}

static inline int kaapi_onereadytasklist_size( const kaapi_onereadytasklist_t* ortl )
{
    return ortl->size;
}


static inline int kaapi_onereadytasklist_pop(
	kaapi_onereadytasklist_t* ortl, kaapi_taskdescr_t** td )
{
    kaapi_atomic_lock( &ortl->lock );
    if( kaapi_onereadytasklist_isempty( ortl ) ){
	kaapi_atomic_unlock( &ortl->lock );
	return EBUSY;
    }
    *td = ortl->head;
    ortl->head = (*td)->next;
    if( ortl->head != NULL )
	ortl->head->prev = 0;
    else
	ortl->tail = NULL; /* empty */
    ortl->size--;
    kaapi_atomic_unlock( &ortl->lock );
    return 0;
}

static inline int kaapi_onereadytasklist_push(
	kaapi_onereadytasklist_t* ortl, kaapi_taskdescr_t* td )
{
    td->next = td->prev = NULL;
    kaapi_atomic_lock( &ortl->lock );
    td->next = ortl->head;
    if( ortl->head != NULL )
	ortl->head->prev = td;
    else
	ortl->tail = td;
    ortl->head = td;
    ortl->size++;
    kaapi_atomic_unlock( &ortl->lock );
    return 0;
}

static inline int kaapi_onereadytasklist_remote_push(
	kaapi_onereadytasklist_t* ortl, kaapi_taskdescr_t* td )
{
    td->next = td->prev = NULL;
    kaapi_atomic_lock( &ortl->lock );
    td->prev = ortl->tail;
    if( ortl->tail != NULL )
	ortl->tail->next = td;
    else
	ortl->head = td;
    ortl->tail = td;
    ortl->size++;
    kaapi_atomic_unlock( &ortl->lock );
    return 0;
}

static inline int kaapi_onereadytasklist_steal(
	kaapi_onereadytasklist_t* ortl, kaapi_taskdescr_t** td ) 
{
    int size_ws;

    kaapi_atomic_lock( &ortl->lock );
    size_ws = kaapi_onereadytasklist_size( ortl );
    if( size_ws == 0 ) {
	kaapi_atomic_unlock( &ortl->lock );
	return 1;
    }
    *td = ortl->tail;
    if( (*td)->prev != NULL )
	(*td)->prev->next = NULL;
    else
	ortl->head = NULL;
    ortl->tail = (*td)->prev;
    ortl->size--;
    kaapi_atomic_unlock( &ortl->lock );
    (*td)->prev = (*td)->next = NULL;
    return 0;
}

/** The workqueue of ready tasks for several priority
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
    if( kaapi_processor_get_type( kaapi_get_current_processor() ) ==
	    KAAPI_PROC_TYPE_CUDA ){
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
  kaapi_readytasklist_t* rtl,
  kaapi_lock_t*          lock
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

static inline int kaapi_readytasklist_count( kaapi_readytasklist_t* rtl )
{
    return KAAPI_ATOMIC_READ( &rtl->cnt_tasks );
}

static inline int kaapi_readylist_push( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t* td, int priority )
{
  kaapi_onereadytasklist_t* ortl;
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MAX_PRIORITY) && (priority <= KAAPI_TASKLIST_MIN_PRIORITY) );

  ortl = &rtl->prl[priority];
  kaapi_onereadytasklist_push( ortl, td );
  KAAPI_ATOMIC_ADD( &rtl->cnt_tasks, 1 );
  return priority;
}

static inline int kaapi_readylist_remote_push(
	kaapi_readytasklist_t* rtl, kaapi_taskdescr_t* td, int priority )
{
  kaapi_onereadytasklist_t* ortl;
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MAX_PRIORITY) && (priority <= KAAPI_TASKLIST_MIN_PRIORITY) );

  ortl = &rtl->prl[priority];
  kaapi_onereadytasklist_remote_push( ortl, td );
  KAAPI_ATOMIC_ADD( &rtl->cnt_tasks, 1 );
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
    for( prio = max_prio; prio != min_prio; prio += inc_prio ) {
	onertl = &rtl->prl[prio];
	err = kaapi_onereadytasklist_steal( onertl, td );
	if( err == 0 ){
	    KAAPI_ATOMIC_ADD( &rtl->cnt_tasks, -1 );
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
    for( prio = max_prio; prio != min_prio; prio += inc_prio ) {
	onertl = &rtl->prl[prio];
	err = kaapi_onereadytasklist_pop( onertl, td );
	if( err == 0 ){
	    KAAPI_ATOMIC_ADD( &rtl->cnt_tasks, -1 );
	    return 0;
	} else
	    if( err != EBUSY )
		return err;
    }
    return EBUSY;
}


#endif /* _KAAPI_READYTASKLIST_H_ */ 
