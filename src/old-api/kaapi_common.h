/*
** kaapi_common.h
** xkaapi
** 
** Created on Tue Mar 31 15:16:12 2009
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
#ifndef _KAAPI_COMMON_TYPE_H
#define _KAAPI_COMMON_TYPE_H 1

#include <stdint.h> /* constant size type definitions */

/** \defgroup TASK Task
    This group defines functions to create and initialize and use task.
*/
/** \defgroup STACK Stack
    This group defines functions to create and initialize stack.
*/

/* ========================================================================= */
/* Kaapi name for stdint typedefs.
 */
typedef uint8_t  kaapi_uint8_t;
typedef uint16_t kaapi_uint16_t;
typedef uint32_t kaapi_uint32_t;
typedef int8_t   kaapi_int8_t;
typedef int16_t  kaapi_int16_t;
typedef int32_t  kaapi_int32_t;

/* ========================================================================= */
/** Thread_t
    The public interface returns only the internal thread identifier (an int)
    and the futur of the result of the thread (!=0 if the thread was not created
    in detached state).
*/
typedef struct kaapi_t {
  int32_t                      tid;
  struct kaapi_thread_futur_t* futur;
} kaapi_t;


#endif /* _KAAPI_COMMON_TYPE_H */
