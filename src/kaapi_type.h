/*
** kaapi_type.h.in
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
#ifndef _KAAPI_TYPE_H
#define _KAAPI_TYPE_H 1

/* =============================== Struct Size ============================== */

#define KAAPI_SIZEOF_MUTEXATTR_T 4

#define KAAPI_SIZEOF_MUTEX_T 68

#define KAAPI_SIZEOF_CONDATTR_T 4

#define KAAPI_SIZEOF_COND_T 60

#define KAAPI_SIZEOF_ATTR_T 24

#define KAAPI_SIZEOF_ONCE_T 4

/* ========================================================================= */
/** Mutex attribut, data structure
 */

typedef struct kaapi_mutexattr_t {
  char _opaque[ KAAPI_SIZEOF_MUTEXATTR_T ]; 
} kaapi_mutexattr_t;

typedef struct kaapi_mutex_t {
  char _opaque[ KAAPI_SIZEOF_MUTEX_T ]; 
} kaapi_mutex_t;

/* ========================================================================= */
/** Condition attribut, data structure
 */
typedef struct kaapi_condattr_t {
  char _opaque[ KAAPI_SIZEOF_CONDATTR_T ];
} kaapi_condattr_t;

typedef struct kaapi_cond_t {
  char _opaque[ KAAPI_SIZEOF_COND_T ];
} kaapi_cond_t;

/* ========================================================================= */
/** Thread attribut
*/
typedef struct kaapi_attr_t {
  char _opaque[ KAAPI_SIZEOF_ATTR_T ]; 
} kaapi_attr_t;
//TODO : initialiazer
//#  define KAAPI_ATTR_INITIALIZER {0, KAAPI_SYSTEM_SCOPE, {~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0}, 0, 0, 0 }

/* ========================================================================= */
/** Once 
*/
typedef struct kaapi_once_t { 
  char _opaque[KAAPI_SIZEOF_ONCE_T];
} kaapi_once_t;

#endif // _KAAPI_TYPE_H
