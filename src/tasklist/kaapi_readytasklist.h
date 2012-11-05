/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
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

struct kaapi_taskdescr_t;

/** One workqueue of ready tasks 
 */
typedef struct kaapi_onereadytasklist_t {
  kaapi_lock_t              lock;
  struct kaapi_taskdescr_t* head;
  struct kaapi_taskdescr_t* tail;
  int                       size;
} kaapi_onereadytasklist_t;

/**/
static inline int kaapi_onereadytasklist_isempty( const kaapi_onereadytasklist_t* ortl )
{
  return ( ortl->size == 0 );
}

static inline int kaapi_onereadytasklist_size( const kaapi_onereadytasklist_t* ortl )
{
  return ortl->size;
}

/** The workqueue of ready tasks with priorities
 */
typedef struct kaapi_readytasklist_t {
  kaapi_onereadytasklist_t prl[KAAPI_TASKLIST_NUM_PRIORITY]; 
  kaapi_atomic_t	         cnt_tasks;
} kaapi_readytasklist_t;

/* 
 * max_prio - maximum priority of a given kproc
 * min_prio - minimum priority of a given kproc
 * inc_prio - increment 
 * Used to iterate from max_prio to min_prio inclusive
 */
static inline void
kaapi_readylist_get_priority_range(
  int* const min_prio, 
  int* const max_prio, 
  int* const inc_prio 
)
{
  *min_prio = KAAPI_TASKLIST_MIN_PRIORITY-1;
  *max_prio = KAAPI_TASKLIST_MAX_PRIORITY;
  *inc_prio = -1;
}

static inline int kaapi_readytasklist_isempty( kaapi_readytasklist_t* rtl )
{
  return (KAAPI_ATOMIC_READ( &rtl->cnt_tasks ) == 0 );
}

static inline int kaapi_readytasklist_workload( kaapi_readytasklist_t* rtl )
{
  return KAAPI_ATOMIC_READ( &rtl->cnt_tasks );
}

/*
 */
extern int kaapi_readytasklist_init( kaapi_readytasklist_t* rtl );

/*
 */
extern int kaapi_readytasklist_destroy( kaapi_readytasklist_t* rtl );


/*
 */
extern int kaapi_readylist_push( kaapi_readytasklist_t* rtl, struct kaapi_taskdescr_t* td );


/*
 */
extern int kaapi_readylist_pop( kaapi_readytasklist_t* rtl, struct kaapi_taskdescr_t** td );

/*
 */
extern int kaapi_readylist_remote_push( kaapi_readytasklist_t* rtl, struct kaapi_taskdescr_t* td );

/*
 */
extern int kaapi_readylist_steal( const struct kaapi_processor_t* thiefprocessor, kaapi_readytasklist_t* rtl, struct kaapi_taskdescr_t** td );

#endif /* _KAAPI_READYTASKLIST_H_ */ 
