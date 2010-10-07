/*
** kaapi_hashvalue.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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
#include "kaapi_impl.h"
#include <string.h>

// --------------------------------------------------------------------
/* source of this function from Paul Hsieh at url
  http://www.azillionmonkeys.com/qed/hash.html
*/
#if !defined (get16bits)
/* [TG] indep from big/little endian. */
#if 1
#define get16bits(d) ( (((const kaapi_uint8_t *)(d))[1] << (kaapi_uint32_t)8)\
                       +((const kaapi_uint8_t *)(d))[0] \
                     )
#else
#  define get16bits(d) ( 0xFFFF & ((const kaapi_uint32_t*)d)[0] )
#endif
#endif

kaapi_uint32_t kaapi_hash_value_len(const char * data, int len) 
{
  if (data == 0) return 0;

  kaapi_uint32_t hash = 0, tmp;
  int rem;

  if (len <= 0) return 0;

  rem = len & 3;
  len >>= 2;

  /* Main loop */
  for (;len > 0; len--) {
      hash  += get16bits (data);
      tmp    = (get16bits (data+2) << 11) ^ hash;
      hash   = (hash << 16) ^ tmp;
      data  += 2*sizeof (kaapi_uint16_t);
      hash  += hash >> 11;
  }

  /* Handle end cases */
  switch (rem) {
      case 3: hash += get16bits (data);
              hash ^= hash << 16;
              hash ^= data[sizeof (kaapi_uint16_t)] << 18;
              hash += hash >> 11;
              break;
      case 2: hash += get16bits (data);
              hash ^= hash << 11;
              hash += hash >> 17;
              break;
      case 1: hash += *data;
              hash ^= hash << 10;
              hash += hash >> 1;
  }

  /* Force "avalanching" of final 127 bits */
  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 2;
  hash += hash >> 15;
  hash ^= hash << 10;

  return hash;
}


kaapi_uint32_t kaapi_hash_ulong(unsigned long value)
{
#define UINT_LOW16_MASK ((1 << 16) - 1)
#define ULONG_LOW32_MASK ((1UL << 32) - 1UL)

  /* low high 32 bits words */
  const kaapi_uint32_t h = (value >> 32) & ULONG_LOW32_MASK;
  const kaapi_uint32_t l = (value >>  0) & ULONG_LOW32_MASK;

  kaapi_uint32_t hash = 0;
  kaapi_uint32_t tmp;

  /* mix high uint32 */
  hash += (h >> 16) & UINT_LOW16_MASK;
  tmp = ((h & UINT_LOW16_MASK) << 11) ^ hash;
  hash = (hash << 16) ^ tmp;
  hash += hash >> 11;

  /* mix low uint32 */
  hash += (l >> 16) & UINT_LOW16_MASK;
  tmp = ((l & UINT_LOW16_MASK) << 11) ^ hash;
  hash = (hash << 16) ^ tmp;
  hash += hash >> 11;

  /* avalanche */
  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 2;
  hash += hash >> 15;
  hash ^= hash << 10;

  return hash;
}


kaapi_uint32_t kaapi_hash_value(const char * data) 
{
  if (data == 0) return 0;

  int len = strlen( data );
  return kaapi_hash_value_len( data, len );
}
