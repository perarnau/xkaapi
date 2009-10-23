/*
** xkaapi
** Created on Tue Mar 31 15:21:08 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
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
** knowledge. Users are therefore encouraged to load and tes t the
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
#ifndef _KAAPI_ADAPT_SYNCHRO_H
#define _KAAPI_ADAPT_SYNCHRO_H

#include "kaapi_config.h"
#include "kaapi_atomic.h"

/**
*/
struct kaapi_s_structure_t;

/** Type of condition that should satisfy dependency
*/
typedef enum kaapi_synchro_op_t {
  KAAPI_OP_EQU = 0,  /* == 0 */
  KAAPI_OP_NEQ = 1,  /* != 0 */
  KAAPI_OP_GEQ = 2,  /* >= 0 */
  KAAPI_OP_GTH = 3,  /* > 0 */
  KAAPI_OP_FUN = 4,  /* f(arg) -> {true|false} */
} kaapi_synchro_op_t;

/** Args for KAAPI_OP_FUN
*/
typedef struct kaapi_synchro_opfunc_t {
  int (*_func)(struct kaapi_s_structure_t*, void*);
  void* _arg;
} kaapi_synchro_opfunc_t;


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

/** Return 1 iff the sstructure is ready depending of the operation
*/
#define kaapi_sstruct_isready_op( kpss, OP, ... )  \
  (OP((kpss)->_status, ##__VA_ARGS__ ))


/** Wait the pis is ready
*/
#define kaapi_sstruct_waitready_eq( kpss, value ) \
  { while (kaapi_sstruct_isready_eq(kpss, value)); kaapi_readmem_barrier(); }

#define kaapi_sstruct_waitready_op( kpss, OP, ... ) \
  { while (kaapi_sstruct_isready_eq(kpss, OP, ##__VA_ARGS__)); kaapi_readmem_barrier(); }


#endif