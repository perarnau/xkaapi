/*
** xkaapi
** 
** Created on Tue Mar 31 15:21:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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
#ifndef _KAAPI__STEALAPI_H_
#define _KAAPI__STEALAPI_H_
#if defined(__cplusplus)
extern "C" {
#endif

#include "kaapi_config.h"
#include "kaapi_error.h"
#include "kaapi_atomic.h"
#include "kaapi_datastructure.h"
#include "kaapi_private_structure.h"
#include "kaapi.h"
#include "kaapi_param.h"
#include "kaapi_stealapi_synchro.h"

#define KAAPI_MAX_PROCESSOR KAAPI_MAX_PROCESSOR

#define KAAPI_STEALCONTEXT_STACKSIZE 4096


/* ========================================================================= */
/** Forward declaration of data structure
*/
struct kaapi_steal_processor_t;  /* where to post a steal request */
struct kaapi_steal_context_t;    /* how to process a steal request */
struct kaapi_steal_request_t;    /* steal request on victim side */


/* ========================================================================= */
/** Forward declaration of global functions
*/
extern void kaapi_stealapi_initialize( );
extern void kaapi_stealapi_terminate();




/* ========================================================================= */
/** This data structure is a context from which steal operations are processed.
    Each control flow may declare a steal context in order to control steal request
    and the protocol to synchronise the thiefs and the victim during the finalization step.
    It is to the responsability of the control flow, either to poll for incomming steal request
    using 'kaapi_stealpoint' or either to mark a section of code that is concurrent with processing
    the steal request (in this case multiple steal requests are processed sequentially).
    The steal context is associated to a stack of bytes used to store the theft context
    and user defined data when a steal request success.
    The stack has a bounded size, if no more space is available for allocation, the
    theft request fails with ENOMEM error code.
*/
typedef int (*kaapi_splitter_t)(
    struct kaapi_steal_processor_t* /*kpss*/, 
    struct kaapi_steal_context_t* /* sc */,
    int /*count*/, 
    kaapi_steal_request_t** /*requests*/
);

typedef struct kaapi_steal_context_t {
  KAAPI_QUEUE_FIELD( struct kaapi_steal_context_t );                           /* to link steal context together in a stack */
  char*                                                       _saved_sp ;      /* saved stack pointer in _stack */
  int volatile                                                _wsreqflag;      /* 0: reject all steal requests
                                                                                  1: non concurrent request
                                                                                  2: concurrent request
                                                                                */
  struct kaapi_steal_processor_t*                             _kprocessor;     /* k-processor owner where I'am pushed */
  kaapi_splitter_t                                            _splitter;       /* !=0 if steal context is pushed or _wsreqflag == 2*/
  kaapi_atomic_t                                              _count_thief;    /* number of stealer of this context */
} kaapi_steal_context_t; 

enum kaapi_steal_context_{
  KAAPI_STEALCONTEXT_S_REQREFUSED = 0,
  KAAPI_STEALCONTEXT_S_REQPOINT   = 1,
  KAAPI_STEALCONTEXT_S_REQCONCUR  = 2
};



/* ========================================================================= */

#define KAAPI_STEAL_PROCESSOR_DECODEINDEX( i ) \
  (i & 0xFF )


#define KAAPI_STEAL_PROCESSOR_SETINDEX( kpsp, index ) \
  ((kpsp)->_processor_id = ((kpsp)->_processor_id & ~0xFF) | index )

/** All available kaapi_steal processor context
*/
extern kaapi_steal_processor_t* kaapi_all_stealprocessor[1+KAAPI_MAX_PROCESSOR];

/** Index for the next stack to declare
*/
extern kaapi_atomic_t kaapi_index_stacksteal;

/** !=0 iff program should terminate
*/
extern int volatile kaapi_stealapi_term;

/* ========================================================================= */
/** Victim part of the protocol
*/

/** Initialize a steal processor context
    Initialize the variable pointed by the parameter
    \param kpss kaapi_steal_processor_t*
    \param index the index of the kaapi_steal_processor_t
    \param sz the size fo the stack
    \param staddr the first address of the stack
*/
int kaapi_steal_processor_init( kaapi_steal_processor_t* kpss, int index, int sz, void* staddr );

/** Terminate a steal processor context.
    Store 0 in the previous place of the stack in kaapi_all_stealprocessor
    Clear all the stack.
    Warning no further synchronization is made to ensure consistency with
    concurrent steal.
    \param kpss the kaapi_steal_processor_t*
*/
int kaapi_steal_processor_terminate( kaapi_steal_processor_t* kpss );


/** Thread entry point for a kernel thread processor
    \param argv should the index of the processor in the stealprocessor global table
    \retval Always return 0
    The function call until terminaison kaapi_sched_select_victim + post request + kaapi_steal_processor.
*/
void* kaapi_steal_processor_run(void* argv);


/** Steal a kaapi_processor
    \param kpss the kaapi_steal_processor_t to steal
    Return the number of sucessfull replies to request
*/
int kaapi_steal_processor( kaapi_steal_processor_t* kpss );

/** Allocate size bytes of memory in a kaapi_steal_processor_t context
    \param kpsc pointer to a steal context
    \param size the size of requested userdata.
    return 0 in case of success
    return ENOMEM if requested space could not be allocated
*/
extern void* kaapi_steal_processor_alloc_data( kaapi_steal_processor_t* kpss, size_t size );

/** Same on the steal context
    extern void* kaapi_steal_context_alloc_data( kaapi_steal_context_t* kpss, size_t size );
*/
#define kaapi_steal_context_alloc_data(kpss, size) \
  kaapi_steal_processor_alloc_data(kpss->_stack, size)

/** Initialize a stealpoint context with size stack size and stackaddr
    \param kpsc a pointer to a kaapi_steal_context_t
    \param sz an unsigned int
    \param staddr a pointer to a memory bloc of size sz bytes
    \retval 0 in case of success
*/
extern int kaapi_steal_context_init( kaapi_steal_context_t* kpsc );

/** Destroy a stealpoint context 
    \param kpsc a pointer to a kaapi_steal_context_t
    \retval 0 in case of success
*/
extern int kaapi_steal_context_destroy( kaapi_steal_context_t* kpsc );

/** Push a stealpoint context into a stealpoint stack
    \param kpscstack a pointer to a kaapi_stealpoint_stack_t
    \param kpsc a pointer to a kaapi_steal_context_t
    \retval 0 in case of success
*/
extern int kaapi_steal_context_push( kaapi_steal_processor_t* kpscstack, kaapi_steal_context_t* kpsc, kaapi_splitter_t splitter );

/** Pop the top of the stealpoint context stack
    \param kpscstack a pointer to a kaapi_stealpoint_stack_t
    \retval 0 in case of success
*/
extern kaapi_steal_context_t* kaapi_steal_context_pop( kaapi_steal_processor_t* kpscstack );

/** Return the top of the stealpoint context stack
    \param kpscstack a pointer to a kaapi_stealpoint_stack_t
*/
#define kaapi_steal_context_top( kpscstack ) \
  KAAPI_QUEUE_FRONT( kpscstack )

/** Define a steal point in the control flow where multiple steal request may be processed.
    If one or serveral request has been posted, then the splitter function is
    called with a count number of posted request and the pointer to the array of requests.
    Not all requests of the array may be valid : the user should iterate through the array
    and test if the request_t is 0 or not. At least count requests are valid when the splitter
    function is called.
    The splitter function should reply to each request using kaapi_request_reply function.
    Warning: On the same steal context, a kaapi_stealpoint could not be nested within 
    a concurrent section defined by  kaapi_with_steal_begin and kaapi_with_steal_end.
    \param kpsc  : is pointer to kaapi_steal_context_t
    \param splitter: the function called to processed the steal. signature: 
                  void ( kaapi_steal_context_t*, int count, kaapi_steal_request_t** request, ... )
    \retval return non zero value in case of steal
*/
#define kaapi_stealpoint( kpsc, splitter, ... ) \
    ( KAAPI_ATOMIC_READ(&((kpsc)->_stack->_list_request._count)) !=0 ? \
         (*splitter)( (kpsc)->_stack, (kpsc), KAAPI_ATOMIC_READ(&((kpsc)->_stack->_list_request._count)), \
                             (kpsc)->_stack->_list_request._request, ##__VA_ARGS__), 1 \
      : 0  \
    )


/** Finalize the control flow
    Then interrupt all thiefs in order to call finalization method on all the kaapi_steal_result_t
    \param kpsc: the stealpoint context
    \param reducer: the function to call after finalization
      should has the signature void (kaapi_stealcontext_t*, void* thief_data, ##__VA_ARGS__)
    CURRENT IMPLEMENTATION IS A FUNCTION, SHOULD BE DEFINED AS MACRO WITH VARIABLE NUMBER OF ARGS
*/
typedef void (*kaapi_reducer_function_t)(kaapi_steal_context_t*, void*, void* /*, void* */);
extern int kaapi_finalize_steal ( 
    kaapi_steal_context_t* kpsc, 
    void* victim_work, 
    kaapi_reducer_function_t fnc_reducer, 
    void* arg_reducer /*, void* arg_reducer2 */
);


/* ========================================================================= */
/** Thief part of the protocol
*/

/** Initialize a request
    \param kpsr a pointer to a kaapi_steal_request_t
*/
#define kaapi_thief_request_init( kpsr, kpss ) \
  (kpsr)->_tpid = kpss->_processor_id;\
  (kpsr)->_entrypoint = 0;\
  kaapi_sstruct_init( kpsr, KAAPI_REQUEST_S_CREATED )

/** Destroy a request
    A posted request could not be destroyed until a reply has been made
*/
#define kaapi_thief_request_destroy( kpsr ) 



/** Return true or false when the request has been processed
  \param pksr kaapi_steal_reply_t
*/
#define kaapi_thief_reply_request( kpsc, array_kpsr, index, flag ) \
{\
    kaapi_steal_request_t* __kpsr = (array_kpsr)[index];\
    (array_kpsr)[index] = 0;\
    if (flag !=0) { \
      __kpsr->_victim_sc = kpsc; \
      kaapi_sstruct_flush_write( __kpsr, KAAPI_REQUEST_S_SUCCESS ); \
      KAAPI_ATOMIC_INCR(&((kpsc)->_count_thief)); \
    }\
    else { \
      kaapi_sstruct_flush_write( __kpsr, KAAPI_REQUEST_S_FAIL );\
    } \
    KAAPI_ATOMIC_DECR(&((kpsc)->_stack->_list_request._count)); \
}


/** Return the processor id of the thief
  \param pksr 
*/
#define kaapi_thief_processor_request( array_kpsr, index ) \
    ((array_kpsr)[index]->_index)



/** Wait the end of request and return the error code
  \param pksr kaapi_steal_reply_t
  \retval KAAPI_REQUEST_S_SUCCESS sucessfull steal operation
  \retval KAAPI_REQUEST_S_FAIL steal request has failed
  \retval KAAPI_REQUEST_S_ERROR steal request has failed to be posted because the victim refused request
  \retval KAAPI_REQUEST_S_QUIT process should terminate
*/
extern int kaapi_thief_request_wait( kaapi_steal_request_t* ksr );


/** Return true iff the request has been processed
  \param pksr kaapi_steal_reply_t
*/
#define kaapi_thief_test_request( kpsr ) \
      (!kaapi_sstruct_isready_eq(kpsr, KAAPI_REQUEST_S_POSTED))

/** Return true iff the request is a success steal
  \param pksr kaapi_steal_reply_t
*/
#define kaapi_thief_request_ok( kpsr ) \
      (kaapi_sstruct_read(kpsr) == KAAPI_REQUEST_S_SUCCESS)

/** Return the request status
  \param pksr kaapi_steal_reply_t
  \retval KAAPI_REQUEST_S_SUCCESS sucessfull steal operation
  \retval KAAPI_REQUEST_S_FAIL steal request has failed
  \retval KAAPI_REQUEST_S_QUIT process should terminate
*/
#define kaapi_thief_request_status( kpsr ) \
      (kaapi_sstruct_read(kpsr))

/**
*/
#define kaapi_thief_execute( kpss, kpsr ) \
  (*(kpsr)->_entrypoint)(kpss, kpsr); \
  KAAPI_ATOMIC_DECR(&(kpsr)->_victim_sc->_count_thief)

#if defined(__cplusplus)
}
#endif

#endif
