/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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
#ifndef _KAAPI_CPUSET_H_
#define _KAAPI_CPUSET_H_ 1

#if defined(__cplusplus)
extern "C" {
#endif

#include "config.h"


/* ========================================================================= */
typedef uint64_t kaapi_cpuset_t[2];


/**
*/
extern const char* kaapi_cpuset2string( int nproc, kaapi_cpuset_t* affinity );


/**
*/
static inline void kaapi_cpuset_clear(kaapi_cpuset_t* affinity )
{
  (*affinity)[0] = 0;
  (*affinity)[1] = 0;
}


/**
*/
static inline void kaapi_cpuset_full(kaapi_cpuset_t* affinity )
{
  (*affinity)[0] = ~0UL;
  (*affinity)[1] = ~0UL;
}


/**
*/
static inline int kaapi_cpuset_intersect(kaapi_cpuset_t* s1, kaapi_cpuset_t* s2)
{
  return (((*s1)[0] & (*s2)[0]) != 0) || (((*s1)[1] & (*s2)[1]) != 0);
}

/**
*/
static inline int kaapi_cpuset_empty(kaapi_cpuset_t* affinity)
{
  return ((*affinity)[0] == 0) && ((*affinity)[1] == 0);
}

/**
*/
static inline int kaapi_cpuset_set(kaapi_cpuset_t* affinity, kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >=0) && (kid < sizeof(kaapi_cpuset_t)*8) );
  if (kid <64)
    (*affinity)[0] |= ((uint64_t)1)<<kid;
  else
    (*affinity)[1] |= ((uint64_t)1)<< (kid-64);
  return 0;
}

/**
*/
static inline int kaapi_cpuset_copy(kaapi_cpuset_t* dest, kaapi_cpuset_t* src )
{
  (*dest)[0] = (*src)[0];
  (*dest)[1] = (*src)[1];
  return 0;
}


/** Return non 0 iff th as affinity with kid
*/
static inline int kaapi_cpuset_has(kaapi_cpuset_t* affinity, kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >=0) && (kid < sizeof(kaapi_cpuset_t)*8) );
  if (kid <64)
    return ( (*affinity)[0] & ((uint64_t)1)<< (uint64_t)kid) != (uint64_t)0;
  else
    return ( (*affinity)[1] & ((uint64_t)1)<< (uint64_t)(kid-64)) != (uint64_t)0;
}

/** Return *dest &= mask
*/
static inline void kaapi_cpuset_and(kaapi_cpuset_t* dest, kaapi_cpuset_t* mask )
{
  (*dest)[0] &= (*mask)[0];
  (*dest)[1] &= (*mask)[1];
}

/** Return *dest |= mask
*/
static inline void kaapi_cpuset_or(kaapi_cpuset_t* dest, kaapi_cpuset_t* mask )
{
  (*dest)[0] |= (*mask)[0];
  (*dest)[1] |= (*mask)[1];
}

/** Return *dest &= ~mask
*/
static inline void kaapi_cpuset_notand(kaapi_cpuset_t* dest, kaapi_cpuset_t* mask )
{
  (*dest)[0] ^= (*mask)[0];
  (*dest)[1] ^= (*mask)[1];
}

#if defined(__cplusplus)
}
#endif

#endif
