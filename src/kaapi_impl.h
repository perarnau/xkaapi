/*
** kaapi_impl.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
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
#ifndef _KAAPI_IMPL_H
#define _KAAPI_IMPL_H 1

#if defined(__cplusplus)
extern "C" {
#endif

/* mark that we compile source of the library */
#define KAAPI_COMPILE_SOURCE 1

#include "kaapi_config.h"
#include "kaapi.h"
#include "kaapi_error.h"

/* Fwd declaration 
*/
struct kaapi_processor_t;
struct kaapi_listrequest_t;

/** Private status of request
    \ingroup WS
*/
enum kaapi_request_status_t {
  KAAPI_REQUEST_S_EMPTY   = 0,
  KAAPI_REQUEST_S_POSTED  = 1,
  KAAPI_REQUEST_S_SUCCESS = 2,
  KAAPI_REQUEST_S_FAIL    = 3,
  KAAPI_REQUEST_S_ERROR   = 4,
  KAAPI_REQUEST_S_QUIT    = 5
};

/** \ingroup WS
    This data structure should contains all necessary informations to post a request to a selected node.
    It should be extended in case of remote work stealing.
*/
typedef struct kaapi_victim_t {
  struct kaapi_processor_t* kproc; /** the victim processor */
  kaapi_uint16_t            level; /** level in the hierarchy of the source k-processor to reach kproc */
} kaapi_victim_t;

/** \ingroup WS
    Select a victim for next steal request
    \param kproc [IN] the kaapi_processor_t that want to emit a request
    \param victim [OUT] the selection of the victim
    \retval 0 in case of success 
    \retval EINTR in case of detection of the termination 
    \retval else error code
    
*/
typedef int (*kaapi_selectvictim_fnc_t)( struct kaapi_processor_t*, struct kaapi_victim_t* );



/** Initialize a request
    \param kpsr a pointer to a kaapi_steal_request_t
*/
#define kaapi_request_init( pkr ) \
  (pkr)->status = KAAPI_REQUEST_S_EMPTY; (pkr)->reply = 0; (pkr)->stack = 0



#include "kaapi_machine.h"


/** Setup KAAPI parameter from
    1/ the command line option
    2/ form the environment variable
    3/ default values
*/
extern int kaapi_setup_param( int argc, char** argv );
    
/** Definition of parameters for the runtime system
*/
typedef struct kaapi_rtparam_t {
  size_t                   stacksize;              /* default stack size */
  unsigned int             syscpucount;            /* number of physical cpus of the system */
  unsigned int             cpucount;               /* number of physical cpu used for execution */
  kaapi_selectvictim_fnc_t wsselect;               /* default method to select a victim */
} kaapi_rtparam_t;

extern kaapi_rtparam_t default_param;



/* ============================= Commun function for server side (no public) ============================ */

/** Useful
*/
#define kaapi_get_current_processor() _kaapi_get_current_processor();

/** \ingroup WS
    Select a victim for next steal request using uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_rand( kaapi_processor_t* kproc, kaapi_victim_t* victim);

/** \ingroup WS
    Select a victim for next steal request using random selection level by level. Each time the method
    try to steal at level i, it first try to steal at level 0 until i.
    The idea of the algorithm is the following. Initial values are: level=0, toplevel=0
       1- the method do random selection at level and increment toplevel
       2- it increment level, if level >= toplevel then level=0, toplevel++
       3- if toplevel > maximal level then level=0, toplevel=0
*/
extern int kaapi_sched_select_victim_rand_incr( kaapi_processor_t* kproc, kaapi_victim_t* victim);

/** \ingroup WS
    Only do rando ws on the first level of the hierarchy. Assume that all cores are connected
    together using the first level hierarchy information.
*/
extern int kaapi_sched_select_victim_rand_first( kaapi_processor_t* kproc, kaapi_victim_t* victim);

/** \ingroup WS
    Helper function for some of the above random selection of victim
*/
extern int kaapi_select_victim_rand_atlevel( kaapi_processor_t* kproc, int level, kaapi_victim_t* victim );


/** \ingroup WS
    Enter in the infinite loop of trying to steal work.
    Never return from this function...
    If proc is null pointer, then the function allocate a new kaapi_processor_t and 
    assigns it to the current processor.
    This method may be called by 'host' thread in order to become executor thread.
    The method returns only in case of terminaison.
*/
extern void kaapi_sched_idle ( kaapi_processor_t* proc );

/** \ingroup WS
    Advance polling of request for the current running thread.
    If this method is called from an other running thread than proc,
    the behavious is unexpected.
    \param proc should be the current running thread
*/
extern int kaapi_sched_advance ( kaapi_processor_t* proc );


/** \ingroup WS
    Splitter for DFG task
    \param proc should be the current running thread
*/
extern int kaapi_task_splitter_dfg(struct kaapi_stack_t* stack, struct kaapi_task_t* task, int count, struct kaapi_request_t* array);



/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/
/* ........................................ PRIVATE INTERFACE ........................................*/
/** \ingroup STACK
    The function kaapi_stack_alloc() allocates in the heap a stack with at most count number of tasks.
    If successful, the kaapi_stack_alloc() function will return zero.  
    Otherwise, an error number will be returned to indicate the error.
    This function is machine dependent.
    \param stack INOUT a pointer to the kaapi_stack_t to allocate.
    \param count_task IN the maximal number of tasks in the stack.
    \param size_data IN the amount of stack data.
    \retval ENOMEM cannot allocate memory.
    \retval EINVAL invalid argument: bad stack pointer or capacity is 0.
*/
extern int kaapi_stack_alloc( kaapi_stack_t* stack, kaapi_uint32_t count_task, kaapi_uint32_t size_data );

/** \ingroup STACK
    The function kaapi_stack_free() free the stack successfuly allocated with kaapi_stack_alloc.
    If successful, the kaapi_stack_free() function will return zero.  
    Otherwise, an error number will be returned to indicate the error.
    This function is machine dependent.
    \param stack INOUT a pointer to the kaapi_stack_t to allocate.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_stack_free( kaapi_stack_t* stack );

/** Post a request to a given k-processor
  This method posts a request to victim k-processor. 
  \param src the sender of the request 
  \param hlevel the hierarchy level of the steal
  \param dest the receiver (victim) of the request
  \param return 0 if the request has been successully posted
  \param return !=0 if the request been not been successully posted and the status of the request contains the error code
*/  
/*extern int kaapi_request_post( kaapi_processor_t* src, kaapi_reply_t* reply, kaapi_victim_t* victim );*/

/** Destroy a request
    A posted request could not be destroyed until a reply has been made
*/
#define kaapi_request_destroy( kpsr ) 


/** Wait the end of request and return the error code
  \param pksr kaapi_reply_t
  \retval KAAPI_REQUEST_S_SUCCESS sucessfull steal operation
  \retval KAAPI_REQUEST_S_FAIL steal request has failed
  \retval KAAPI_REQUEST_S_ERROR steal request has failed to be posted because the victim refused request
  \retval KAAPI_REQUEST_S_QUIT process should terminate
*/
extern int kaapi_reply_wait( kaapi_reply_t* ksr );

/** Return true iff the request has been posted
  \param pksr kaapi_request_t
*/
static inline int kaapi_request_test( kaapi_request_t* kpsr )
{ return (kpsr->status == KAAPI_REQUEST_S_POSTED); }

/** Return true iff the request has been processed
  \param pksr kaapi_reply_t
*/
static inline int kaapi_reply_test( kaapi_reply_t* kpsr )
{ return (kpsr->status != KAAPI_REQUEST_S_POSTED); }

/** Return true iff the request is a success steal
  \param pksr kaapi_reply_t
*/
static inline int kaapi_reply_ok( kaapi_reply_t* kpsr )
{ return (kpsr->status == KAAPI_REQUEST_S_SUCCESS); }

/** Return the request status
  \param pksr kaapi_reply_t
  \retval KAAPI_REQUEST_S_SUCCESS sucessfull steal operation
  \retval KAAPI_REQUEST_S_FAIL steal request has failed
  \retval KAAPI_REQUEST_S_QUIT process should terminate
*/
static inline int kaapi_request_status( kaapi_reply_t* reply ) 
{ return reply->status; }

/** Return the data associated with the reply
  \param pksr kaapi_reply_t
*/
static inline kaapi_stack_t* kaapi_request_data( kaapi_reply_t* reply ) 
{ 
  kaapi_readmem_barrier();
  return reply->data; 
}

/** Return true if the whole application is terminated 
*/
/*extern int kaapi_isterminated(void);*/

/** Here: kaapi_atomic_t + functions ; termination detecting barrier, see mt_machine.h
*/


/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/
/* ........................................ PUBLIC INTERFACE ........................................*/


#endif /* _KAAPI_IMPL_H */
