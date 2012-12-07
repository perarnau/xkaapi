/*
** xkaapi
** 
**
** Copyright 2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** Vincent.Danjean@ens-lyon.org
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
#include <stdio.h>
#include "kaapic.h"

#ifndef T
#  error T not defined
#endif
#define KAAPIC_TYPE(s) _KAAPIC_TYPE(s)
#define _KAAPIC_TYPE(s) KAAPIC_TYPE_##s

void body(T const n1, T const n2, T const n3, T const n4,
	  T const n5, T const n6, T const n7, T const n8,
	  T* result)
{
#define PTR 1
#if KT == PTR
  *result = (void*)((uintptr_t)n1 + (uintptr_t)n2 + (uintptr_t)n3 + (uintptr_t)n4
		+ (uintptr_t)n5 + (uintptr_t)n6 + (uintptr_t)n7 + (uintptr_t)n8);
#else
  *result = n1+n2+n3+n4+n5+n6+n7+n8;
#endif
#undef PTR
}


int main()
{
  T result;
  kaapic_spawn_attr_t attr;
  
  kaapic_init(KAAPIC_START_ONLY_MAIN);

  kaapic_begin_parallel (KAAPIC_FLAG_DEFAULT);

  /* initialize default attribut */
  kaapic_spawn_attr_init(&attr);
  
  /* spawn the task */
  kaapic_spawn(&attr, 
      9,                /* number of arguments */
      (void(*)())body,  /* the entry point for the task */
#if 0
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)32,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)32,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)32,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)16,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)8,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)4,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)2,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)1,
      KAAPIC_MODE_W, KAAPIC_TYPE(KT), 1, &result
#else
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)0x10,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)0x20,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)0x30,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)0x40,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)0x50,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)0x60,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)0x70,
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)0x80,
      KAAPIC_MODE_W, KAAPIC_TYPE(KT), 1, &result
#endif
  );

  kaapic_sync();
  
  kaapic_end_parallel (KAAPIC_FLAG_DEFAULT);

  printf("The result is : %i/%f\n", (int)(uintptr_t)result, (double)(uintptr_t)result );
  kaapic_finalize();

#if 0
  if (result == (T)127) {
    printf("Success\n");
  }
#else
  if (result == (T)((T)0x10+(T)0x20+(T)0x30+(T)0x40+(T)0x50+(T)0x60+(T)0x70+(T)0x80) {
    printf("Success\n");
  }
#endif  
  return 0;
}
