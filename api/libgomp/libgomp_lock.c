/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** francois.broquedis@imag.fr
** vincent.danjean@imag.fr
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

/* We want to use our lock definition */
#define _LIBGOMP_OMP_LOCK_DEFINED 1

#include "lock-internal.h"

#include "libgomp.h"

/* Simple lock: map omp_lock_t to the smallest atomic in kaapi...
   Cannot reuse kaapi_lock_t because Kaapi data structure
*/
void komp_init_lock_30(omp_lock_t *lock)
{
  KAAPI_ATOMIC_WRITE(lock, 1);
}

void komp_destroy_lock_30(omp_lock_t *lock)
{
  kaapi_assert_debug(KAAPI_ATOMIC_READ(lock)==1);
}

void komp_set_lock_30(omp_lock_t *lock)
{
acquire:
  if (KAAPI_ATOMIC_DECR(lock) ==0) 
    return;
  while (KAAPI_ATOMIC_READ(lock) <=0)
    kaapi_slowdown_cpu();
  goto acquire;
}

void komp_unset_lock_30(omp_lock_t *lock)
{
  kaapi_mem_barrier();
  KAAPI_ATOMIC_WRITE(lock, 1);
}

int komp_test_lock_30(omp_lock_t *lock)
{
  if ((KAAPI_ATOMIC_READ(lock) ==1) && (KAAPI_ATOMIC_DECR(lock) ==0))
    return 1;
  return 0;
}

#if defined(HAVE_VERSION_SYMBOL)
strong_alias (komp_init_lock_30, komp_init_lock_25)
strong_alias (komp_destroy_lock_30, komp_destroy_lock_25)
strong_alias (komp_set_lock_30, komp_set_lock_25)
strong_alias (komp_unset_lock_30, komp_unset_lock_25)
strong_alias (komp_test_lock_30, komp_test_lock_25)

komp_lock_symver(omp_init_lock)
komp_lock_symver(omp_destroy_lock)
komp_lock_symver(omp_set_lock)
komp_lock_symver(omp_unset_lock)
komp_lock_symver(omp_test_lock)
#endif

/* nested lock */
void komp_init_nest_lock_30(omp_nest_lock_t *lock)
{
  komp_init_lock_30( &lock->lock );
  lock->count = 0;
  lock->owner = 0;
}

void komp_destroy_nest_lock_30(omp_nest_lock_t *lock)
{
  komp_destroy_lock_30(&lock->lock);
}

void komp_set_nest_lock_30(omp_nest_lock_t *lock)
{
  void *me = (void*)(intptr_t)komp_get_ctxt();
  if (lock->owner != me) 
  {
    komp_set_lock_30(&lock->lock);
    lock->owner = me;
  }
  lock->count++;
}

void komp_unset_nest_lock_30(omp_nest_lock_t *lock)
{
  if (--lock->count == 0) {
    lock->owner = NULL;
    komp_unset_lock_30(&lock->lock);
  }
}

int komp_test_nest_lock_30(omp_nest_lock_t *lock)
{
  void *me = (void*)(intptr_t)komp_get_ctxt();
  if (lock->owner == me) 
  {
    lock->count++;
    return lock->count;
  }
  if (komp_test_lock_30(&lock->lock)) 
  {
    lock->owner=me;
    lock->count=1;
    return 1;
  }
  return 0;
}

#if defined(HAVE_VERSION_SYMBOL)
komp_lock_symver30(omp_init_nest_lock)
komp_lock_symver30(omp_destroy_nest_lock)
komp_lock_symver30(omp_set_nest_lock)
komp_lock_symver30(omp_unset_nest_lock)
komp_lock_symver30(omp_test_nest_lock)
#endif
