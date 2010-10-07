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

  On the web page, 2010-10-06:
  "IMPORTANT NOTE: Since there has been a lot of interest for the code below, 
  I have decided to additionally provide it under the GPL 2.0 license. 
  This provision applies to the code below only and not to any other code including 
  other source archives or listings from this site unless otherwise specified. 

  The GPL 2.0 is not necessarily a more liberal license than my derivative license, 
  but this additional licensing makes the code available to more developers. 
  Note that this does not give you multi-licensing rights. 
  You can only use the code under one of the licenses at a time."
*/
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#  define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ( (((const kaapi_uint8_t *)(d))[1] << (kaapi_uint32_t)8)\
                       +((const kaapi_uint8_t *)(d))[0] \
                     )
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

kaapi_uint32_t kaapi_hash_value(const char * data) 
{
  if (data == 0) return 0;

  int len = strlen( data );
  return kaapi_hash_value_len( data, len );
}
