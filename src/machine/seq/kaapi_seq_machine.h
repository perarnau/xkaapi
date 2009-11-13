/*
** kaapi_seq_machine.h
** xkaapi
** 
** Created on Tue Mar 31 15:20:42 2009
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
#ifndef _KAAPI_MACHINE_H_
#define _KAAPI_MACHINE_H_ 1

/* THIS FILE IS INCLUDE WITH kaapi_impl.h: kaapi_task_t and kaapi_stack_t will be defined */


/* ============================= Kprocessor ============================ */
/** This data structure defines a work stealer processor thread
    \ingroup WS
*/
typedef struct kaapi_processor_t {
  kaapi_uint32_t           kid;            /* Kprocessor id */
  kaapi_stack_t            stack;          /* attached stack */
} kaapi_processor_t;


extern kaapi_processor_t* kaapi_main_processor; /* the current processor in sequential version */

/** \ingroup WS
    Returns the current stack of tasks
*/
#define kaapi_self_stack() \
(&(kaapi_main_processor->stack))


/* ============================= Scheduler machine dependent ============================ */
extern void kaapi_sched_idle    ( kaapi_processor_t* proc );
extern int kaapi_sched_advance  ( kaapi_processor_t* proc );



/* ============================= Stack machine dependent ============================ */
extern int kaapi_stack_alloc( kaapi_stack_t* stack, kaapi_uint32_t count_task, kaapi_uint32_t size_data );
extern int kaapi_stack_free( kaapi_stack_t* stack );



/* ============================= Internal machine dependent routines ============================ */
/**
*/
extern kaapi_thread_id_t kaapi_allocate_tid(int type);


/* ============================= Initialization routines ============================ */
/** Function called before main entry point in order to initialize the library
*/
void __attribute__ ((constructor)) kaapi_init(void)

/** Function called afer the main entry point in order to initialize the library
*/
void __attribute__ ((destructor)) kaapi_term(void)

/* ============================= Atomic Function ============================ */

#  define KAAPI_ATOMIC_CAS(a, o, n) \
 (*(a) == o ? (*(a)=n, 1) : 0)

#  define KAAPI_ATOMIC_INCR(a) \
  ++(a)->_counter

#  define KAAPI_ATOMIC_DECR(a) \
  --(a)->_counter

#  define KAAPI_ATOMIC_ADD(a, value) \
  (a)->_counter += value

#  define KAAPI_ATOMIC_SUB(a, value) \
  (a)->_counter -= value

#  define KAAPI_ATOMIC_READ(a) \
  ((a)->_counter)

#  define KAAPI_ATOMIC_WRITE(a, value) \
  (a)->_counter = value

/* ========================================================================== */

/* ============================= Memory Barrier ============================= */

#define kaapi_writemem_barrier() \
  /* nothing */

#define void kaapi_readmem_barrier()  \
  /* nothing */


/* ========================================================================== */
/** Termination dectecting barrier
*/
/**
*/
#define kaapi_barrier_td_init( kpb, value ) \
  KAAPI_ATOMIC_WRITE( kpb, value )

/**
*/
#define kaapi_barrier_td_destroy( kpb )
  
/**
*/
#define kaapi_barrier_td_setactive( kpb, b ) \
  if (b) { KAAPI_ATOMIC_INCR( kpb ); } \
  else KAAPI_ATOMIC_DECR( kpb )

/**
*/
#define kaapi_barrier_td_isterminated( kpb ) \
  (KAAPI_ATOMIC_READ(kpb ) == 0)



/* ========================================================================== */
/** Communication structure buffer between thread
*/
struct kaapi_s_structure_t;

/** The size of data part for structure for synchronisation
*/
enum { KAAPI_SSTRUCTURE_DATA_MAX= (KAAPI_CACHE_LINE-sizeof(int)) };

#define KAAPI_INHERITE_FROM_SSTRUCT_T \
  int _status

typedef struct kaapi_s_structure_t {
  KAAPI_INHERITE_FROM_SSTRUCT_T;
  char  _data[KAAPI_SSTRUCTURE_DATA_MAX];
} kaapi_s_structure_t;


/** Initialize the struct for synchronization with given type
    \param kpss should have the _status field defined by KAAPI_INHERITE_SSTRUCT_T
*/
#define kaapi_sstruct_init( kpss, value ) \
  (kpss)->_status =value

/** Destroy the struct for synchronisation
*/
#define kaapi_sstruct_destroy( kpss )

/** Flush write operation on the structure and write the status
    \param kpss should have the _status field defined by KAAPI_INHERITE_SSTRUCT_T
*/
#define kaapi_sstruct_flush_write( kpss, value ) \
    kaapi_writemem_barrier();\
    (kpss)->_status = value

/** Write the status
    \param kpss should have the _status field defined by KAAPI_INHERITE_SSTRUCT_T
*/
#define kaapi_sstruct_write( kpss, value ) \
    (kpss)->_status = value


/** Read the status
    \param kpss should have the _status field defined by KAAPI_INHERITE_SSTRUCT_T
*/
#define kaapi_sstruct_read( kpss ) \
    (kpss)->_status


/** Return 1 iff the sstructure is ready depending of the operation
*/
#define kaapi_sstruct_isready_eq( kpss, value )  \
  ((kpss)->_status == value)


/** Wait the pis is ready
*/
#define kaapi_sstruct_waitready_eq( kpss, value ) \
  { while (kaapi_sstruct_isready_eq(kpss, value)); kaapi_readmem_barrier(); }

#endif /* _KAAPI_ATOMIC_H */
