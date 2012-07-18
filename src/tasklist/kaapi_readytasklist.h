
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
    kaapi_atomic_initlock( &ortl->lock ); /* TODO: use same lock from kproc ? */
}

static inline void kaapi_onereadytasklist_destroy( kaapi_onereadytasklist_t* ortl )
{
    ortl->size = 0;
    ortl->head = ortl->tail = NULL;
    kaapi_atomic_destroylock( &ortl->lock ); /* TODO: use same lock from kproc ? */
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
} kaapi_readytasklist_t;


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
    return 0;
}

static inline int kaapi_readytasklist_destroy( kaapi_readytasklist_t* rtl )
{
    int i;

    for (i =0; i<KAAPI_TASKLIST_NUM_PRIORITY; ++i) 
	kaapi_onereadytasklist_destroy( &rtl->prl[i] );
    return 0;
}


/** Activate and push ready tasks of an activation link.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
*/
static inline int kaapi_readylist_push( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t* td, int priority )
{
  kaapi_onereadytasklist_t* ortl;
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MAX_PRIORITY) && (priority <= KAAPI_TASKLIST_MIN_PRIORITY) );

  ortl = &rtl->prl[priority];
  kaapi_onereadytasklist_push( ortl, td );
#if 0
  if( td->fmt != 0 )
      fprintf(stdout, "[%s] kid=%lu pushed td=%p prio=%d name=%s (counter=%d,wc=%d)\n", 
	      __FUNCTION__,
		(long unsigned int)kaapi_get_current_kid(),
	      (void*)td, priority, td->fmt->name,
	      KAAPI_ATOMIC_READ(&td->counter),
	      td->wc
	      );
  else
      fprintf(stdout, "[%s] kid=%lu pushed td=%p prio=%d (counter=%d,wc=%d)\n", 
	      __FUNCTION__,
		(long unsigned int)kaapi_get_current_kid(),
	      (void*)td, priority,
	      KAAPI_ATOMIC_READ(&td->counter),
	      td->wc
	     );
  fflush(stdout);
#endif
  return priority;
}

/** Return the number of tasks inside the tasklist
*/
static inline size_t kaapi_readylist_workload( kaapi_readytasklist_t* rtl )
{
  size_t size = 0;
  kaapi_onereadytasklist_t* onertl;
  int i;

  for (i =KAAPI_TASKLIST_MAX_PRIORITY; i<(1+KAAPI_TASKLIST_MIN_PRIORITY); ++i)
  {
    onertl = &rtl->prl[i];
    size = 100*size+kaapi_onereadytasklist_size( onertl );
  }
  return size;
}

static inline int kaapi_readylist_steal( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** td ) 
{
    kaapi_onereadytasklist_t* onertl;
    int err;
    int prio;

    for( prio= KAAPI_TASKLIST_MAX_PRIORITY; prio < (1+KAAPI_TASKLIST_MIN_PRIORITY); ++prio ){
	onertl = &rtl->prl[prio];
	err = kaapi_onereadytasklist_steal( onertl, td );
	if( err == 0 ){
#if 0
	  if( (*td)->fmt != 0 )
	      fprintf(stdout, "[%s] kid=%lu pushed td=%p prio=%d name=%s (counter=%d,wc=%d)\n", 
		      __FUNCTION__,
			(long unsigned int)kaapi_get_current_kid(),
		      (void*)*td, prio, (*td)->fmt->name,
		      KAAPI_ATOMIC_READ(&(*td)->counter),
		      (*td)->wc
		      );
	  else
	      fprintf(stdout, "[%s] kid=%lu pushed td=%p prio=%d (counter=%d,wc=%d)\n", 
		      __FUNCTION__,
			(long unsigned int)kaapi_get_current_kid(),
		      (void*)*td, prio,
		      KAAPI_ATOMIC_READ(&(*td)->counter),
		      (*td)->wc
		     );
	  fflush(stdout);
#endif
	    return 0;
	}
    }
    return 1;
}

/** To pop the next ready tasks
    Return the next task to execute if err ==0
    Return 0 if case of success
    Return EBUSY if the workqueue is empty
    Else return an error code
*/
static inline int kaapi_readylist_pop( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** td )
{
  kaapi_onereadytasklist_t* onertl;
  int prio;
  int err;

  for (prio =KAAPI_TASKLIST_MAX_PRIORITY; prio<(1+KAAPI_TASKLIST_MIN_PRIORITY); ++prio) {
    onertl = &rtl->prl[prio];
    err = kaapi_onereadytasklist_pop( onertl, td );
    if( err == 0 ){
#if 0
	  if( (*td)->fmt != 0 )
	      fprintf(stdout, "[%s] kid=%lu pushed td=%p prio=%d name=%s (counter=%d,wc=%d)\n", 
		      __FUNCTION__,
			(long unsigned int)kaapi_get_current_kid(),
		      (void*)*td, prio, (*td)->fmt->name,
		      KAAPI_ATOMIC_READ(&(*td)->counter),
		      (*td)->wc
		      );
	  else
	      fprintf(stdout, "[%s] kid=%lu pushed td=%p prio=%d (counter=%d,wc=%d)\n", 
		      __FUNCTION__,
			(long unsigned int)kaapi_get_current_kid(),
		      (void*)*td, prio,
		      KAAPI_ATOMIC_READ(&(*td)->counter),
		      (*td)->wc
		     );
	  fflush(stdout);
#endif
	return 0;
    }else
	if( err != EBUSY )
	    return err;
  }
  return EBUSY;
}


#endif /* _KAAPI_READYTASKLIST_H_ */ 
